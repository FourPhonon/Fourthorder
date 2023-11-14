# Fourthorder #


Zherui Han <zrhan@purdue.edu>
Xiaolong Yang <xiaolongyang1990@gmail.com>
Wu Li <wu.li.phys2011@gmail.com>
Tianli Feng <Tianli.Feng2011@gmail.com>
Xiulin Ruan <ruan@purdue.edu>


The Fourthorder scripts were derived from the thirdorder scripts, which are used to help users of [ShengBTE containing four-phonon modules] create FORCE\_CONSTANTS\_4TH files in an efficient and convenient manner. More specifically, it performs two tasks:

1) It resolves an irreducible set of atomic displacements from which to compute the full anharmonic fourth-order interatomic force constant (IFC) matrix. The displaced supercells are saved to input files that can be fed to first-principles DFT codes for calculating the forces arising from the atomic displacements. Currently supported DFT codes are VASP (Fourthorder_vasp.py), and Quantum ESPRESSO (Fourthorder_espresso.py).

2) From the output files created by the DFT code, Fourthorder reconstructs the full fourth-order IFC matrix and writes it in the right format to FORCE\_CONSTANTS\_4TH.

*We suggest that users test the convergence of 4th-IFCs with respect to the finite displacement (H). You can modify the header of Fourthorder_common.py to manually change H.*

# Compilation #

Fourthorder is a set of Python scripts. It was developed using Python 2.7.3, but should work with slightly older versions. In addition to the modules in Python's standard library, the numpy and scipy numerical libraries are required. Moreover, this script relies on a module, Fourthorder\_core, which is written in Cython. Thus, in spite of Python being an interpreted language, a compilation step is needed. Note that in addition to the .pyx source we also distribute the intermediate .c file, so Cython itself is not needed. The only requirements are a C compiler, the Python development package and Atsushi Togo's [spglib](http://spglib.sourceforge.net/).

Compiling can be as easy as running

```bash
./compile.sh
```

However, if you have installed spglib to a nonstandard directory, you will have to perform some simple editing on setup.py so that the compiler can find it. Please refer to the comments in that file.

# Usage #

After a successful compilation, the directory will contain the compiled module Fourthorder\_core.so, Fourthorder_common.py and DFT-code specific interfaces (e.g. Fourthorder_vasp.py). All are needed to run Fourthorder. You can either use them from that directory (maybe including it in your PATH for convenience) or copying the .py files to a directory in your PATH and Fourthorder\_core.so to any location where Python can find it for importing.

# Running Fourthorder with VASP #

Any invocation of Fourthorder_vasp.py requires a POSCAR file with a description of the unit cell to be present in the current directory. The script uses no other configuration files, and takes exactly five mandatory command-line arguments:

```bash
Fourthorder_vasp.py sow|reap na nb nc cutoff[nm/-integer]
```

The first argument must be either "sow" or "reap", and chooses the operation to be performed (displacement generation or IFC matrix reconstruction). The next three must be positive integers, and specify the dimensions of the supercell to be created. Finally, the "cutoff" parameter decides on a force cutoff distance. Interactions between atoms spaced further than this parameter are neglected. If cutoff is a positive real number, it is interpreted as a distance in nm; on the other hand, if it is a negative integer -n, the maximum distance among n-th neighbors in the supercell is automatically determined and the cutoff distance is set accordingly.

The following POSCAR describes the relaxed geometry of the primitive unit cell of InAs, a III-V semiconductor with a zincblende structure:

```
InAs
   6.00000000000000
     0.0000000000000000    0.5026468896190005    0.5026468896190005
     0.5026468896190005    0.0000000000000000    0.5026468896190005
     0.5026468896190005    0.5026468896190005    0.0000000000000000
   In   As
   1   1
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
```

Let us assume that such POSCAR is in the current directory and that Fourthorder_vasp.py is in our PATH. To generate an irreducible set of displacements for a 4x4x4 supercell and up-to-second-neighbor interactions, we run

```bash
Fourthorder_vasp.py sow 4 4 4 -2
```

This creates a file called 4TH.SPOSCAR with the undisplaced supercell coordinates and 744 files with names following the pattern 4TH.POSCAR.*. It is the latter that need to be input to VASP. This step is completely system-dependent, but suppose that in ~/vaspinputs we have the required INCAR, POTCAR and KPOINTS files as well as a runvasp.sh script that can be passed to qsub. We can run a command sequence like:

```bash
for i in 4TH.POSCAR.*;do
   s=$(echo $i|cut -d"." -f3) &&
   d=job-$s &&
   mkdir $d &&
   cp $i $d/POSCAR &&
   cp ~/vaspinputs/INCAR ~/vaspinputs/POTCAR ~/vaspinputs/KPOINTS $d &&
   cp ~/vaspinputs/runvasp.sh $d &&
   (cd $d && qsub runvasp.sh)
done
```

Some time later, after all these jobs have finished successfully, we only need to feed all the vasprun.xml files in the right order to Fourthorder_vasp.py, this time in reap mode:

```bash
find job* -name vasprun.xml|sort -n|Fourthorder_vasp.py reap 4 4 4 -2
```

If everything goes according to plan, a FORCE\_CONSTANTS\_4TH file will be created at the end of this run. Naturally, it is important to choose the same parameters for the sow and reap steps.

# Running Fourthorder with Quantum ESPRESSO #

The invocation of Fourthorder_espresso.py requires two files:

1) an input file of the unit cell with converged structural parameters

2) a template input file for the supercell calculations. The template file is a normal QE input file with some wildcards

The following input file GaAs.in describes the relaxed geometry of the primitive unit cell of GaAs, a III-V semiconductor with a zincblende structure

```
&CONTROL
 calculation='scf',
 prefix='gaas',
 restart_mode='from_scratch',
 tstress = .true.,
 tprnfor = .true.,
/
&SYSTEM
 ibrav=0,
 nat=2,
 ntyp=2,
 ecutwfc=48
 ecutrho=384
/
&ELECTRONS
 conv_thr=1.d-12,
/
ATOMIC_SPECIES
 Ga  69.723    Ga.pbe-dnl-kjpaw_psl.1.0.0.UPF
 As  74.92160  As.pbe-dn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 Ga 0.00 0.00 0.00
 As 0.25 0.25 0.25
K_POINTS automatic
11 11 11 0 0 0
CELL_PARAMETERS angstrom
   0.000000000   2.857507756   2.857507756
   2.857507756   0.000000000   2.857507756
   2.857507756   2.857507756   0.000000000
```

Fourthorder_espresso.py supports the following QE input conventions for structural parameters:

1) ibrav = 0 together with CELL_PARAMETERS (alat | bohr | angstrom)

2) ibrav != 0 together with celldm(1)-celldm(6)

For ATOMIC_POSITIONS, all QE units are supported (alat | bohr | angstrom | crystal). Simple algebraic expressions for the positions are supported in similar fashion to QE. Cases ibrav = -5, -9, -12, -13, and 91 are not currently implemented (but those structures can be defined via ibrav = 0 instead)

The following supercell template GaAs_sc.in is used for creating the supercell input files (note the ##WILDCARDS##):

```
&CONTROL
  calculation='scf',
  prefix='gaas',
  tstress = .true.,
  tprnfor = .true.,
  outdir = 'tmp_##NUMBER##'
/
&SYSTEM
  ibrav=0,
  nat=##NATOMS##,
  ntyp=2,
  ecutwfc=48
  ecutrho=384
/
&ELECTRONS
  conv_thr=1.d-12,
/
ATOMIC_SPECIES
 As  74.92160  As.pbe-dn-kjpaw_psl.1.0.0.UPF
 Ga  69.723    Ga.pbe-dnl-kjpaw_psl.1.0.0.UPF
##COORDINATES##
K_POINTS gamma
##CELL##
```

Please note that if Gamma-point k-sampling is used for the supercells, it is computationally much more efficient to apply "K_POINTS gamma" instead of "K_POINTS automatic" with the mesh set to "1 1 1 0 0 0". SCF convergence criterion conv_thr should be set to a tight value and parameters tstress and tprnfor are required so that Fourthorder can extract the forces from the output file.

Fourthorder uses no other configuration files, and requires seven mandatory command-line arguments to create the supercell inputs with the "sow" operation:

```bash
Fourthorder_espresso.py unitcell.in sow na nb nc cutoff[nm/-integer] supercell_template.in
```

Please see the above description for VASP for the explanation of the parameters na, nb, nc, and cutoff. For the present GaAs example, we execute:

```bash
Fourthorder_espresso.py GaAs.in sow 4 4 4 -2 GaAs_sc.in
```

The command creates a file called BASE.GaAs_sc.in with the undisplaced supercell coordinates and 744 files with names following the pattern DISP.GaAs_sc.in.NNN The DISP files should be executed with QE. This step is completely system-dependent, but some practical suggestions can be extracted from the VASP example above.

After all the jobs have finished successfully, we only need to feed all the output files in the right order to Fourthorder_espresso.py, this time in reap mode (now using only six arguments, the supercell argument is not used here):

```bash
find . -name 'DISP.GaAs_sc.in*out' | sort -n | Fourthorder_espresso.py GaAs.in reap 4 4 4 -2
```

If everything goes according to plan, a FORCE_CONSTANTS_4TH file will be created at the end of this run. Naturally, it is important to choose the same parameters for the sow and reap steps.
