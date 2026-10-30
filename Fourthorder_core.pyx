#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Cython fourth-order core using the fifthorder_cython reduction strategy.

import itertools
import os
import sys

import numpy as np
cimport cython
cimport numpy as cnp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import spglib
from libc.math cimport fabs

cnp.import_array()

try:
    xrange
except NameError:
    xrange = range

ORDER = 4
NCOMP = 3 ** ORDER
PERMUTATIONS = tuple(itertools.permutations(range(ORDER)))
DIRECTION_TUPLES = tuple(itertools.product(range(3), repeat=ORDER))
DIR_ARRAY = np.array(DIRECTION_TUPLES, dtype=np.intc)
EPS = 1e-10
ENFORCE_SITE_SYMMETRY = os.environ.get(
    "FOURTHORDER_ENFORCE_SITE_SYMMETRY", "1") == "1"
ASR_MODES = ("none", "weak", "strict", "strict-all")


def _ind2id(icell, ispecies, ngrid, nspecies):
    return ((icell[0] + (icell[1] + icell[2] * ngrid[1]) * ngrid[0])
            * nspecies + ispecies)


def _id2ind(ngrid, nspecies):
    ntot = int(ngrid[0] * ngrid[1] * ngrid[2] * nspecies)
    icell = np.empty((3, ntot), dtype=np.intc)
    ispecies = np.empty(ntot, dtype=np.intc)
    for ii in xrange(ntot):
        tmp, ispecies[ii] = divmod(ii, nspecies)
        tmp, icell[0, ii] = divmod(tmp, ngrid[0])
        icell[2, ii], icell[1, ii] = divmod(tmp, ngrid[1])
    return icell, ispecies


def _dataset_get(dataset, *names):
    for name in names:
        if hasattr(dataset, name):
            return getattr(dataset, name)
        try:
            return dataset[name]
        except (KeyError, TypeError):
            pass
    raise KeyError(names[0])


class SymmetryOperations(object):
    def __init__(self, lattvec, types, positions, symprec=1e-5):
        self.__lattvec = np.array(lattvec, dtype=np.double)
        self.__types = np.array(types, dtype=np.intc)
        self.__positions = np.array(positions, dtype=np.double)
        if self.__positions.shape[1] != 3:
            raise ValueError("positions must be a natoms x 3 array")
        if self.__lattvec.shape != (3, 3):
            raise ValueError("lattice vectors must form a 3 x 3 matrix")
        self.natoms = self.__positions.shape[0]
        self.symprec = symprec
        dataset = spglib.get_symmetry_dataset(
            (self.__lattvec.T, self.__positions, self.__types),
            symprec=symprec)
        if dataset is None:
            raise RuntimeError("spglib could not determine symmetry")
        self.symbol = str(_dataset_get(
            dataset, "international", "international_symbol")).strip()
        self.__rotations = np.array(_dataset_get(dataset, "rotations"),
                                    dtype=np.double)
        self.__translations = np.array(_dataset_get(dataset, "translations"),
                                       dtype=np.double)
        self.nsyms = self.__rotations.shape[0]
        self.__shift = np.zeros(3, dtype=np.double)
        self.__transform = np.eye(3, dtype=np.double)
        inv_lattvec = scipy.linalg.inv(self.__lattvec)
        self.__crotations = np.empty_like(self.__rotations, dtype=np.double)
        self.__ctranslations = np.empty_like(self.__translations,
                                             dtype=np.double)
        for i in xrange(self.nsyms):
            self.__crotations[i, :, :] = np.dot(
                self.__lattvec, np.dot(self.__rotations[i, :, :], inv_lattvec))
            self.__ctranslations[i, :] = np.dot(self.__lattvec,
                                                self.__translations[i, :])

    @property
    def lattice_vectors(self):
        return self.__lattvec

    @property
    def types(self):
        return self.__types

    @property
    def positions(self):
        return self.__positions

    @property
    def origin_shift(self):
        return self.__shift

    @property
    def transformation_matrix(self):
        return self.__transform

    @property
    def rotations(self):
        return self.__rotations

    @property
    def translations(self):
        return self.__translations

    @property
    def crotations(self):
        return self.__crotations

    @property
    def ctranslations(self):
        return self.__ctranslations

    def map_supercell(self, sposcar):
        positions = sposcar["positions"]
        lattvec = sposcar["lattvec"]
        ngrid = np.array([sposcar["na"], sposcar["nb"], sposcar["nc"]],
                         dtype=np.intc)
        ntot = positions.shape[1]
        natoms = ntot // int(np.prod(ngrid))
        motif = np.dot(self.__lattvec, self.__positions.T)
        nruter = np.empty((self.nsyms, ntot), dtype=np.intc)
        factorization = scipy.linalg.lu_factor(self.__lattvec)
        for atom in xrange(ntot):
            car = np.dot(lattvec, positions[:, atom])
            for isym in xrange(self.nsyms):
                car_sym = (np.dot(self.__crotations[isym], car)
                           + self.__ctranslations[isym])
                for iat in xrange(natoms):
                    tmp = scipy.linalg.lu_solve(factorization,
                                                car_sym - motif[:, iat])
                    vec = np.rint(tmp).astype(np.intc)
                    diff = np.abs(vec - tmp).sum()
                    vec %= ngrid
                    if diff < 1e-4:
                        nruter[isym, atom] = _ind2id(vec, iat, ngrid, natoms)
                        break
                else:
                    sys.exit("Error: equivalent atom not found for isym={}, atom={}".
                             format(isym, atom))
        return nruter


def _rotation_matrix(rot_cart, permutation):
    matrix = np.ones((NCOMP, NCOMP), dtype=np.double)
    permuted_inputs = DIR_ARRAY[:, permutation]
    for axis in xrange(ORDER):
        matrix *= rot_cart[np.ix_(DIR_ARRAY[:, axis],
                                  permuted_inputs[:, axis])]
    matrix[np.abs(matrix) < 1e-12] = 0.0
    return matrix


cdef double EPS_C = 1e-10


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef tuple _gaussian(double[:, :] a):
    """
    Specialized version of Gaussian elimination for the 3^4 component space.
    """
    cdef int i, j, k, irow, pivot
    cdef int row_count = a.shape[0]
    cdef int dependent_count = 0
    cdef int independent_count = 0
    cdef double best, tmp
    dependent_arr = np.empty(NCOMP, dtype=np.intc)
    independent_arr = np.empty(NCOMP, dtype=np.intc)
    b = np.zeros((NCOMP, NCOMP), dtype=np.double)
    cdef int[:] dependent = dependent_arr
    cdef int[:] independent = independent_arr
    cdef double[:, :] b_view = b

    irow = 0
    for k in range(NCOMP):
        for i in range(row_count):
            if fabs(a[i, k]) < EPS_C:
                a[i, k] = 0.0
        if irow < row_count - 1:
            pivot = irow
            best = fabs(a[irow, k])
            for i in range(irow + 1, row_count):
                if fabs(a[i, k]) > best:
                    pivot = i
                    best = fabs(a[i, k])
            if fabs(a[pivot, k]) - fabs(a[irow, k]) > EPS_C:
                for j in range(k, NCOMP):
                    tmp = a[irow, j]
                    a[irow, j] = a[pivot, j]
                    a[pivot, j] = tmp
        if fabs(a[irow, k]) > EPS_C:
            dependent[dependent_count] = k
            dependent_count += 1
            for j in range(NCOMP - 1, k, -1):
                a[irow, j] /= a[irow, k]
            a[irow, k] = 1.0
            for i in range(row_count):
                if i == irow:
                    continue
                for j in range(NCOMP - 1, k, -1):
                    a[i, j] -= a[i, k] * a[irow, j] / a[irow, k]
                a[i, k] = 0.0
            if irow < row_count - 1:
                irow += 1
        else:
            independent[independent_count] = k
            independent_count += 1
    for j in range(independent_count):
        for i in range(dependent_count):
            b_view[dependent[i], j] = -a[i, independent[j]]
        b_view[independent[j], j] = 1.0
    return b[:, :independent_count], independent_arr[:independent_count]


def gaussian(rows):
    row_count = max(len(rows), NCOMP)
    a = np.zeros((row_count, NCOMP), dtype=np.double)
    for i, row in enumerate(rows):
        a[i, :] = row
    return _gaussian(a)


class Wedge(object):
    def __init__(self, poscar, sposcar, symops, dmin, nequis, shifts, frange):
        self.poscar = poscar
        self.sposcar = sposcar
        self.symops = symops
        self.dmin = dmin
        self.nequis = nequis
        self.shifts = shifts
        self.frange = frange
        self.nequi = []
        self.llist = []
        self.allequilist = []
        self.nindependentbasis = []
        self.independentbasis = []
        self.transformationarray = []
        self._reduce()
        self.nlist = len(self.llist)
        self.nequi = np.array(self.nequi, dtype=np.intc)
        self.nindependentbasis = np.array(self.nindependentbasis,
                                          dtype=np.intc)

    def _normalize_quartet(self, quartet, isym, id_equi, ind_cell, ind_species,
                           ngrid, natoms):
        mapped = [int(id_equi[isym, atom]) for atom in quartet]
        vecs = [ind_cell[:, atom].copy() for atom in mapped]
        if np.any(vecs[0] != 0):
            origin = vecs[0].copy()
            for i in xrange(ORDER):
                vecs[i] = (vecs[i] - origin) % ngrid
        result = []
        for i in xrange(ORDER):
            species = int(ind_species[mapped[i]])
            result.append(_ind2id(vecs[i], species, ngrid, natoms))
        return tuple(result)

    def _cluster_is_in_range(self, atoms, shift_options, frange2):
        lattvec = self.sposcar["lattvec"]
        positions = self.sposcar["positions"]
        best_d2 = np.inf
        for shifts in itertools.product(*shift_options):
            cars = [np.dot(lattvec, np.asarray(shift) + positions[:, atom])
                    for atom, shift in zip(atoms, shifts)]
            max_d2 = 0.0
            for a, b in itertools.combinations(range(len(cars)), 2):
                diff = cars[a] - cars[b]
                d2 = float(np.dot(diff, diff))
                if d2 > max_d2:
                    max_d2 = d2
                    if max_d2 >= best_d2:
                        break
            if max_d2 < best_d2:
                best_d2 = max_d2
                if best_d2 < frange2:
                    return True
        return best_d2 < frange2

    def _reduce(self):
        frange2 = self.frange * self.frange
        ngrid = np.array([self.sposcar["na"], self.sposcar["nb"],
                          self.sposcar["nc"]], dtype=np.intc)
        natoms = len(self.poscar["types"])
        ntot = len(self.sposcar["types"])
        shifts27 = np.array(list(itertools.product(range(-1, 2), repeat=3)),
                            dtype=np.intc)
        id_equi = self.symops.map_supercell(self.sposcar)
        ind_cell, ind_species = _id2ind(ngrid, natoms)
        all_seen = set()

        for ii in xrange(natoms):
            for jj in xrange(ntot):
                if self.dmin[ii, jj] >= self.frange:
                    continue
                shift2all = [tuple(shifts27[idx])
                             for idx in self.shifts[ii, jj,
                                                    :self.nequis[ii, jj]]]
                for kk in xrange(ntot):
                    if self.dmin[ii, kk] >= self.frange:
                        continue
                    shift3all = [tuple(shifts27[idx])
                                 for idx in self.shifts[ii, kk,
                                                        :self.nequis[ii, kk]]]
                    for ll in xrange(ntot):
                        if self.dmin[ii, ll] >= self.frange:
                            continue
                        shift4all = [tuple(shifts27[idx])
                                     for idx in self.shifts[ii, ll,
                                                            :self.nequis[ii, ll]]]
                        if not self._cluster_is_in_range(
                                (jj, kk, ll),
                                (shift2all, shift3all, shift4all),
                                frange2):
                            continue
                        quartet = (ii, jj, kk, ll)
                        if quartet in all_seen:
                            continue
                        equilist = []
                        local_seen = set()
                        transforms = []
                        coeff_rows = []
                        for permutation in PERMUTATIONS:
                            permuted = tuple(quartet[p] for p in permutation)
                            for isym in xrange(self.symops.nsyms):
                                quartet_sym = self._normalize_quartet(
                                    permuted, isym, id_equi, ind_cell,
                                    ind_species, ngrid, natoms)
                                add_equiv = (
                                    (permutation == PERMUTATIONS[0]
                                     and isym == 0) or
                                    (quartet_sym != quartet
                                     and quartet_sym not in local_seen))
                                need_rot = (add_equiv or
                                            (ENFORCE_SITE_SYMMETRY and
                                             quartet_sym == quartet))
                                if not need_rot:
                                    continue
                                rot = _rotation_matrix(
                                    self.symops.crotations[isym],
                                    permutation)
                                if add_equiv:
                                    local_seen.add(quartet_sym)
                                    all_seen.add(quartet_sym)
                                    equilist.append(quartet_sym)
                                    transforms.append(rot)
                                if (ENFORCE_SITE_SYMMETRY and
                                        quartet_sym == quartet):
                                    diff = rot - np.eye(NCOMP)
                                    nz = np.where(np.any(np.abs(diff) > 1e-12,
                                                        axis=1))[0]
                                    coeff_rows.extend(diff[row, :]
                                                     for row in nz)
                        if ENFORCE_SITE_SYMMETRY:
                            b, independent = gaussian(coeff_rows)
                        else:
                            b = np.eye(NCOMP, dtype=np.double)
                            independent = np.arange(NCOMP, dtype=np.intc)
                        transarray = np.empty((len(transforms), NCOMP,
                                               len(independent)),
                                              dtype=np.double)
                        for idx, rot in enumerate(transforms):
                            transarray[idx, :, :] = np.dot(rot, b)
                            transarray[idx,
                                       np.abs(transarray[idx]) < 1e-12] = 0.0
                        self.llist.append(quartet)
                        self.nequi.append(len(equilist))
                        self.allequilist.append(equilist)
                        self.nindependentbasis.append(len(independent))
                        self.independentbasis.append(independent)
                        self.transformationarray.append(transarray)

    def build_list4(self):
        """Return unique 6-tuples: three displaced atoms followed by directions."""
        nruter = []
        seen = set()
        for class_index, quartet in enumerate(self.llist):
            for basis_index in self.independentbasis[class_index]:
                dirs = DIRECTION_TUPLES[int(basis_index)]
                task = (quartet[0], quartet[1], quartet[2],
                        dirs[0], dirs[1], dirs[2])
                if task not in seen:
                    seen.add(task)
                    nruter.append(task)
        return nruter

    def build_list6(self):
        return self.build_list4()


def _format_relative(value, scale):
    if scale <= 0.0:
        return 0.0
    return value / scale


def _asr_matrix_weak(wedge, equiv_lookup, naccum, natoms, ntot, nparams):
    rows = []
    cols = []
    vals = []
    ncols = natoms * ntot * NCOMP
    colindex = 0
    for ii in xrange(natoms):
        for jj in xrange(ntot):
            for comp in xrange(NCOMP):
                for kk in xrange(ntot):
                    for ll in xrange(ntot):
                        found = equiv_lookup.get((ii, jj, kk, ll))
                        if found is None:
                            continue
                        class_index, equiv_index = found
                        offset = naccum[class_index]
                        coeffs = wedge.transformationarray[class_index][
                            equiv_index, comp, :]
                        nz = np.nonzero(np.abs(coeffs) > 1e-12)[0]
                        rows.extend((offset + nz).tolist())
                        cols.extend([colindex] * len(nz))
                        vals.extend(coeffs[nz].tolist())
                colindex += 1
    if not vals:
        return None
    return scipy.sparse.coo_matrix(
        (vals, (rows, cols)), (nparams, ncols)).tocsr()


def _append_constraint_terms(row_terms, wedge, naccum, found, comp):
    class_index, equiv_index = found
    offset = naccum[class_index]
    coeffs = wedge.transformationarray[class_index][equiv_index, comp, :]
    nz = np.nonzero(np.abs(coeffs) > 1e-12)[0]
    for idx in nz:
        col = int(offset + idx)
        row_terms[col] = row_terms.get(col, 0.0) + float(coeffs[idx])


def _flush_constraint_row(row_terms, row_index, rows, cols, vals):
    wrote = False
    for col, value in row_terms.items():
        if abs(value) > 1e-12:
            rows.append(row_index)
            cols.append(col)
            vals.append(value)
            wrote = True
    return wrote


def _asr_matrix_strict(wedge, equiv_lookup, naccum, natoms, ntot, nparams,
                       axes):
    rows = []
    cols = []
    vals = []
    row_index = 0
    for axis in axes:
        if axis == 3:
            for ii in xrange(natoms):
                for jj in xrange(ntot):
                    for kk in xrange(ntot):
                        for comp in xrange(NCOMP):
                            row_terms = {}
                            for ll in xrange(ntot):
                                found = equiv_lookup.get((ii, jj, kk, ll))
                                if found is not None:
                                    _append_constraint_terms(
                                        row_terms, wedge, naccum, found, comp)
                            if _flush_constraint_row(
                                    row_terms, row_index, rows, cols, vals):
                                row_index += 1
        elif axis == 2:
            for ii in xrange(natoms):
                for jj in xrange(ntot):
                    for ll in xrange(ntot):
                        for comp in xrange(NCOMP):
                            row_terms = {}
                            for kk in xrange(ntot):
                                found = equiv_lookup.get((ii, jj, kk, ll))
                                if found is not None:
                                    _append_constraint_terms(
                                        row_terms, wedge, naccum, found, comp)
                            if _flush_constraint_row(
                                    row_terms, row_index, rows, cols, vals):
                                row_index += 1
        elif axis == 1:
            for ii in xrange(natoms):
                for kk in xrange(ntot):
                    for ll in xrange(ntot):
                        for comp in xrange(NCOMP):
                            row_terms = {}
                            for jj in xrange(ntot):
                                found = equiv_lookup.get((ii, jj, kk, ll))
                                if found is not None:
                                    _append_constraint_terms(
                                        row_terms, wedge, naccum, found, comp)
                            if _flush_constraint_row(
                                    row_terms, row_index, rows, cols, vals):
                                row_index += 1
        else:
            raise ValueError("invalid ASR axis")
    if not vals:
        return None
    return scipy.sparse.coo_matrix(
        (vals, (rows, cols)), (row_index, nparams)).tocsr()


def _apply_weak_asr(aphilist, wedge, equiv_lookup, naccum, natoms, ntot):
    aa = _asr_matrix_weak(
        wedge, equiv_lookup, naccum, natoms, ntot, aphilist.size)
    if aa is None:
        print("- Weak ASR: no constraints inside the cutoff")
        return aphilist
    residual_before = aa.T.dot(aphilist)
    max_before = float(np.max(np.abs(residual_before)))
    norm_before = float(np.linalg.norm(residual_before))
    print("- Weak ASR matrix: {0} x {1}, nnz={2}".format(
        aa.shape[0], aa.shape[1], aa.nnz))
    print("- Weak ASR residual before: max={0:.6e}, norm={1:.6e}".format(
        max_before, norm_before))
    D = scipy.sparse.spdiags(
        aphilist, [0], aphilist.size, aphilist.size, format="csr")
    bbs = D.dot(aa)
    multiplier = -scipy.sparse.linalg.lsqr(
        bbs, np.ones_like(aphilist))[0]
    correction = D.dot(bbs.dot(multiplier))
    corrected = aphilist + correction
    residual_after = aa.T.dot(corrected)
    max_after = float(np.max(np.abs(residual_after)))
    norm_after = float(np.linalg.norm(residual_after))
    denom = max(float(np.linalg.norm(aphilist)), EPS)
    rel = float(np.linalg.norm(correction)) / denom
    print("- Weak ASR residual after:  max={0:.6e}, norm={1:.6e}".format(
        max_after, norm_after))
    print("- Weak ASR relative correction norm: {0:.6e}".format(rel))
    return corrected


def _apply_strict_asr(aphilist, wedge, equiv_lookup, naccum, natoms, ntot,
                      mode):
    if mode == "strict":
        axes = (3,)
    else:
        axes = (1, 2, 3)
    cc = _asr_matrix_strict(
        wedge, equiv_lookup, naccum, natoms, ntot, aphilist.size, axes)
    if cc is None:
        print("- Strict ASR: no constraints inside the cutoff")
        return aphilist
    residual_before = cc.dot(aphilist)
    max_before = float(np.max(np.abs(residual_before)))
    norm_before = float(np.linalg.norm(residual_before))
    print("- Strict ASR matrix: {0} x {1}, nnz={2}".format(
        cc.shape[0], cc.shape[1], cc.nnz))
    print("- Strict ASR residual before: max={0:.6e}, norm={1:.6e}".format(
        max_before, norm_before))
    if max_before < 1e-14:
        print("- Strict ASR relative correction norm: 0.000000e+00")
        return aphilist
    max_param = float(np.max(np.abs(aphilist))) if aphilist.size else 0.0
    if max_param <= 0.0:
        weights = np.ones_like(aphilist)
    else:
        weights = np.maximum(np.abs(aphilist), 1e-6 * max_param)
    scale_matrix = scipy.sparse.spdiags(
        weights, [0], aphilist.size, aphilist.size, format="csr")
    weighted_cc = cc.dot(scale_matrix)
    result = scipy.sparse.linalg.lsmr(
        weighted_cc, -residual_before, atol=1e-12, btol=1e-12)
    correction = weights * result[0]
    corrected = aphilist + correction
    residual_after = cc.dot(corrected)
    max_after = float(np.max(np.abs(residual_after)))
    norm_after = float(np.linalg.norm(residual_after))
    denom = max(float(np.linalg.norm(aphilist)), EPS)
    rel = float(np.linalg.norm(correction)) / denom
    print("- Strict ASR residual after:  max={0:.6e}, norm={1:.6e}".format(
        max_after, norm_after))
    print("- Strict ASR relative correction norm: {0:.6e}".format(rel))
    print("- Strict ASR solver: istop={0}, iterations={1}, normr={2:.6e}".
          format(result[1], result[2], result[3]))
    return corrected


def _print_asr_diagnostics(phifull):
    scale = float(np.max(np.abs(phifull))) if phifull.size else 0.0
    sum_j = float(np.max(np.abs(phifull.sum(axis=5)))) if phifull.size else 0.0
    sum_k = float(np.max(np.abs(phifull.sum(axis=6)))) if phifull.size else 0.0
    sum_l = float(np.max(np.abs(phifull.sum(axis=7)))) if phifull.size else 0.0
    sum_kl = float(np.max(np.abs(phifull.sum(axis=(6, 7))))) if phifull.size else 0.0
    print("- ASR diagnostics on reconstructed IFCs")
    print("- \t max |Phi|: {0:.6e}".format(scale))
    print("- \t max |sum_j Phi|: {0:.6e} ({1:.6e} relative)".format(
        sum_j, _format_relative(sum_j, scale)))
    print("- \t max |sum_k Phi|: {0:.6e} ({1:.6e} relative)".format(
        sum_k, _format_relative(sum_k, scale)))
    print("- \t max |sum_l Phi|: {0:.6e} ({1:.6e} relative)".format(
        sum_l, _format_relative(sum_l, scale)))
    print("- \t max |sum_kl Phi|: {0:.6e} ({1:.6e} relative)".format(
        sum_kl, _format_relative(sum_kl, scale)))


def reconstruct_ifcs(phipart, wedge, list4, poscar, sposcar, asr_mode="weak",
                     asr_diagnostics=True):
    if asr_mode not in ASR_MODES:
        raise ValueError("invalid ASR mode: {0}".format(asr_mode))
    natoms = len(poscar["types"])
    ntot = len(sposcar["types"])
    phifull = np.zeros((3, 3, 3, 3, natoms, ntot, ntot, ntot),
                       dtype=np.double)
    for ii, entry in enumerate(list4):
        e0, e1, e2, e3, e4, e5 = entry
        phifull[e3, e4, e5, :, e0, e1, e2, :] = phipart[:, ii, :]

    naccum = np.insert(np.cumsum(wedge.nindependentbasis, dtype=np.intc), 0, 0)
    philist = []
    for class_index, quartet in enumerate(wedge.llist):
        for basis_index in wedge.independentbasis[class_index]:
            dirs = DIRECTION_TUPLES[int(basis_index)]
            philist.append(phifull[dirs + quartet])
    aphilist = np.array(philist, dtype=np.double)

    equiv_lookup = {}
    for class_index, equilist in enumerate(wedge.allequilist):
        for equiv_index, quartet in enumerate(equilist):
            equiv_lookup[quartet] = (class_index, equiv_index)

    print("- ASR mode: {0}".format(asr_mode))
    if aphilist.size and asr_mode == "weak":
        aphilist = _apply_weak_asr(
            aphilist, wedge, equiv_lookup, naccum, natoms, ntot)
    elif aphilist.size and asr_mode in ("strict", "strict-all"):
        aphilist = _apply_strict_asr(
            aphilist, wedge, equiv_lookup, naccum, natoms, ntot, asr_mode)

    phifull[:, :, :, :, :, :, :, :] = 0.0
    for class_index, equilist in enumerate(wedge.allequilist):
        offset = naccum[class_index]
        nind = wedge.nindependentbasis[class_index]
        params = aphilist[offset:offset + nind]
        for equiv_index, quartet in enumerate(equilist):
            values = np.dot(wedge.transformationarray[class_index][equiv_index],
                            params)
            values[np.abs(values) < 1e-12] = 0.0
            for comp, value in enumerate(values):
                if value != 0.0:
                    phifull[DIRECTION_TUPLES[comp] + quartet] = value
    if asr_diagnostics:
        _print_asr_diagnostics(phifull)
    return phifull
