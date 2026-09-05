Epitoscope extension
####################

PocketScope is a repo about finding pockets in proteins.

I have extended the idea for a tool that tells me, for a collection of pdb files, which epitopes would be most suitable to target with an antibody (e.g. a nanobody), so that there is minimal cross-talk between each nanobody when trying to bind their proper antigen, if there are other antigens in solution for it.

As part of this epitope selection, I also wanted to take into account that the proteins in each pdb file might be difficult to express or to keep folded in the correct conformation in an vitro transcription and translation system, so we should concentrate on the parts of the proteins that would have a the best chance to fold correctly in such IVTT system, similarly how many of the PDB structures are not for the full conformation for the protein, but exclude parts of the protein that are difficult to crystalize or run on Cryo-EM.

The initial version of this tool is as a python tool, and all the dependencies contained can either be in a python venv environment or a conda environment, with instructions on how to install the dependencies.

