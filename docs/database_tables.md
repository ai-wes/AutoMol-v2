BIOTHERAPEUTIC_COMPONENTS:
Links each biotherapeutic drug (in the biotherapeutics table) to its component sequences (in the bio_component_sequences table). A biotherapeutic drug can have multiple components and hence multiple rows in this table. Similarly, a particular component sequence can be part of more than one drug.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK BIOCOMP_ID NUMBER NOT NULL Primary key.
FK,UK MOLREGNO NUMBER NOT NULL Foreign key to the biotherapeutics table, indicating which biotherapeutic the component is part of.
FK,UK COMPONENT_ID NUMBER NOT NULL Foreign key to the bio_component_sequences table, indicating which component is part of the biotherapeutic.

BIOTHERAPEUTICS:
Table mapping biotherapeutics (e.g. recombinant proteins, peptides and antibodies) to the molecule_dictionary table. Includes HELM notation where available.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK,FK MOLREGNO NUMBER NOT NULL Foreign key to molecule_dictionary
DESCRIPTION VARCHAR2(2000) Description of the biotherapeutic.
HELM_NOTATION VARCHAR2(4000) Sequence notation generated according to the HELM standard (http://www.openhelm.org/home). Currently for peptides only

CHEMBL_ID_LOOKUP:
Lookup table storing chembl identifiers for different entities in the database (assays, compounds, documents and targets)

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK CHEMBL_ID VARCHAR2(20) NOT NULL ChEMBL identifier
UK ENTITY_TYPE VARCHAR2(50) NOT NULL Type of entity (e.g., COMPOUND, ASSAY, TARGET)
UK ENTITY_ID NUMBER NOT NULL Primary key for that entity in corresponding table (e.g., molregno for compounds, tid for targets)
STATUS VARCHAR2(10) NOT NULL Indicates whether the status of the entity within the database - ACTIVE, INACTIVE (downgraded), OBS (obsolete/removed).
LAST_ACTIVE NUMBER indicates the last ChEMBL version where the CHEMBL_ID was active

COMPONENT_CLASS:
Links protein components of targets to the protein_family_classification table. A protein can have more than one classification (e.g., Membrane receptor and Enzyme).

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
FK,UK COMPONENT_ID NUMBER NOT NULL Foreign key to component_sequences table.
FK,UK PROTEIN_CLASS_ID NUMBER NOT NULL Foreign key to the protein_classification table.
PK COMP_CLASS_ID NUMBER NOT NULL Primary key.

COMPONENT_DOMAINS:
Links protein components of targets to the structural domains they contain (from the domains table). Contains information showing the start and end position of the domain in the component sequence.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK COMPD_ID NUMBER NOT NULL Primary key.
FK,UK DOMAIN_ID NUMBER Foreign key to the domains table, indicating the domain that is contained in the associated molecular component.
FK,UK COMPONENT_ID NUMBER NOT NULL Foreign key to the component_sequences table, indicating the molecular_component that has the given domain.
UK START_POSITION NUMBER Start position of the domain within the sequence given in the component_sequences table.
END_POSITION NUMBER End position of the domain within the sequence given in the component_sequences table.

COMPONENT_SEQUENCES:
Table storing the sequences for components of molecular targets (e.g., protein sequences), along with other details taken from sequence databases (e.g., names, accessions). Single protein targets will have a single protein component in this table, whereas protein complexes/protein families will have multiple protein components.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK COMPONENT_ID NUMBER NOT NULL Primary key. Unique identifier for the component.
COMPONENT_TYPE VARCHAR2(50) Type of molecular component represented (e.g., 'PROTEIN','DNA','RNA').
UK ACCESSION VARCHAR2(25) Accession for the sequence in the source database from which it was taken (e.g., UniProt accession for proteins).
SEQUENCE CLOB A representative sequence for the molecular component, as given in the source sequence database (not necessarily the exact sequence used in the assay).
SEQUENCE_MD5SUM VARCHAR2(32) MD5 checksum of the sequence.
DESCRIPTION VARCHAR2(200) Description/name for the molecular component, usually taken from the source sequence database.
TAX_ID NUMBER NCBI tax ID for the sequence in the source database (i.e., species that the protein/nucleic acid sequence comes from).
ORGANISM VARCHAR2(150) Name of the organism the sequence comes from.
DB_SOURCE VARCHAR2(25) The name of the source sequence database from which sequences/accessions are taken. For UniProt proteins, this field indicates whether the sequence is from SWISS-PROT or TREMBL.
DB_VERSION VARCHAR2(10) The version of the source sequence database from which sequences/accession were last updated.

COMPONENT_SYNONYMS:
Table storing synonyms for the components of molecular targets (e.g., names, acronyms, gene symbols etc.) Please note: EC numbers are also currently included in this table although they are not strictly synonyms and can apply to multiple proteins.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK COMPSYN_ID NUMBER NOT NULL Primary key.
FK,UK COMPONENT_ID NUMBER NOT NULL Foreign key to the component_sequences table. The component to which this synonym applies.
UK COMPONENT_SYNONYM VARCHAR2(500) The synonym for the component.
UK SYN_TYPE VARCHAR2(20) The type or origin of the synonym (e.g., GENE_SYMBOL).

DOMAINS:
Table storing a non-redundant list of domains found in protein targets (e.g., Pfam domains).

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK DOMAIN_ID NUMBER NOT NULL Primary key. Unique identifier for each domain.
DOMAIN_TYPE VARCHAR2(20) NOT NULL Indicates the source of the domain (e.g., Pfam).
SOURCE_DOMAIN_ID VARCHAR2(20) NOT NULL Identifier for the domain in the source database (e.g., Pfam ID such as PF00001).
DOMAIN_NAME VARCHAR2(100) Name given to the domain in the source database (e.g., 7tm_1).
DOMAIN_DESCRIPTION VARCHAR2(500) Longer name or description for the domain.
DRUG_MECHANISM:
Table storing mechanism of action information for drugs, and clinical candidate drugs, from a variety of sources (e.g., ATC, FDA, ClinicalTrials.gov)

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK MEC_ID NUMBER NOT NULL Primary key for each drug mechanism of action
FK RECORD_ID NUMBER NOT NULL Record_id for the drug (foreign key to compound_records table)
FK MOLREGNO NUMBER Molregno for the drug (foreign key to molecule_dictionary table)
MECHANISM_OF_ACTION VARCHAR2(250) Description of the mechanism of action e.g., 'Phosphodiesterase 5 inhibitor'
FK TID NUMBER Target associated with this mechanism of action (foreign key to target_dictionary table)
FK SITE_ID NUMBER Binding site for the drug within the target (where known) - foreign key to binding_sites table
FK ACTION_TYPE VARCHAR2(50) Type of action of the drug on the target e.g., agonist/antagonist etc (foreign key to action_type table)
DIRECT_INTERACTION NUMBER Flag to show whether the molecule is believed to interact directly with the target (1 = yes, 0 = no)
MOLECULAR_MECHANISM NUMBER Flag to show whether the mechanism of action describes the molecular target of the drug, rather than a higher-level physiological mechanism e.g., vasodilator (1 = yes, 0 = no)
DISEASE_EFFICACY NUMBER Flag to show whether the target assigned is believed to play a role in the efficacy of the drug in the indication(s) for which it is approved (1 = yes, 0 = no)
MECHANISM_COMMENT VARCHAR2(2000) Additional comments regarding the mechanism of action
SELECTIVITY_COMMENT VARCHAR2(1000) Additional comments regarding the selectivity of the drug
BINDING_SITE_COMMENT VARCHAR2(1000) Additional comments regarding the binding site of the drug
FK VARIANT_ID NUMBER Foreign key to variant_sequences table. Indicates the mutant/variant version of the target used in the assay (where known/applicable)

DRUG_WARNING:
Table storing safety-related information for drugs and clinical candidates

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK WARNING_ID NUMBER NOT NULL Primary key for the drug warning
FK RECORD_ID NUMBER Foreign key to the compound_records table
MOLREGNO NUMBER Foreign key to molecule_dictionary table
WARNING_TYPE VARCHAR2(20) Description of the drug warning type (e.g., withdrawn vs black box warning)
WARNING_CLASS VARCHAR2(100) High-level class of the drug warning
WARNING_DESCRIPTION VARCHAR2(4000) Description of the drug warning
WARNING_COUNTRY VARCHAR2(1000) List of countries/regions associated with the drug warning
WARNING_YEAR NUMBER Earliest year the warning was applied to the drug.
EFO_TERM VARCHAR2(200) Term for Experimental Factor Ontology (EFO)
EFO_ID VARCHAR2(20) Identifier for Experimental Factor Ontology (EFO)
EFO_ID_FOR_WARNING_CLASS VARCHAR2(20) Warning Class Identifier for Experimental Factor Ontology (EFO)

METABOLISM:
Table storing drug metabolic pathways, manually curated from a variety of sources

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK MET_ID NUMBER NOT NULL Primary key
FK,UK DRUG_RECORD_ID NUMBER Foreign key to compound_records. Record representing the drug or other compound for which metabolism is being studied (may not be the same as the substrate being measured)
FK,UK SUBSTRATE_RECORD_ID NUMBER Foreign key to compound_records. Record representing the compound that is the subject of metabolism
FK,UK METABOLITE_RECORD_ID NUMBER Foreign key to compound_records. Record representing the compound that is the result of metabolism
UK PATHWAY_ID NUMBER Identifier for the metabolic scheme/pathway (may be multiple pathways from one source document)
PATHWAY_KEY VARCHAR2(50) Link to original source indicating where the pathway information was found (e.g., Figure 1, page 23)
UK ENZYME_NAME VARCHAR2(200) Name of the enzyme responsible for the metabolic conversion
FK,UK ENZYME_TID NUMBER Foreign key to target_dictionary. TID for the enzyme responsible for the metabolic conversion
MET_CONVERSION VARCHAR2(200) Description of the metabolic conversion
ORGANISM VARCHAR2(100) Organism in which this metabolic reaction occurs
UK TAX_ID NUMBER NCBI Tax ID for the organism in which this metabolic reaction occurs
MET_COMMENT VARCHAR2(1000) Additional information regarding the metabolism (e.g., organ system, conditions under which observed, activity of metabolites)

METABOLISM_REFS:
Table storing references for metabolic pathways, indicating the source of the data

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK METREF_ID NUMBER NOT NULL Primary key
FK,UK MET_ID NUMBER NOT NULL Foreign key to record_metabolism table - indicating the metabolism information to which the references refer
UK REF_TYPE VARCHAR2(50) NOT NULL Type/source of reference (e.g., 'PubMed','DailyMed')
UK REF_ID VARCHAR2(200) Identifier for the reference in the source (e.g., PubMed ID or DailyMed setid)
REF_URL VARCHAR2(400) Full URL linking to the reference
PREDICTED_BINDING_DOMAINS:
Table storing information on the likely binding domain of compounds in the activities table (based on analysis of the domain structure of the target. Note these are predictions, not experimentally determined. See Kruger F, Rostom R and Overington JP (2012), BMC Bioinformatics, 13(S17), S11 for more details.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK PREDBIND_ID NUMBER NOT NULL Primary key.
FK ACTIVITY_ID NUMBER Foreign key to the activities table, indicating the compound/assay(+target) combination for which this prediction is made.
FK SITE_ID NUMBER Foreign key to the binding_sites table, indicating the binding site (domain) that the compound is predicted to bind to.
PREDICTION_METHOD VARCHAR2(50) The method used to assign the binding domain (e.g., 'Single domain' where the protein has only 1 domain, 'Multi domain' where the protein has multiple domains, but only 1 is known to bind small molecules in other proteins).
CONFIDENCE VARCHAR2(10) The level of confidence assigned to the prediction (high where the protein has only 1 domain, medium where the compound has multiple domains, but only 1 known small molecule-binding domain).
PROTEIN_CLASS_SYNONYMS:
Table storing synonyms for the protein family classifications (from various sources including MeSH, ConceptWiki and UMLS).

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK PROTCLASSSYN_ID NUMBER NOT NULL Primary key.
FK,UK PROTEIN_CLASS_ID NUMBER NOT NULL Foreign key to the PROTEIN_CLASSIFICATION table. The protein_class to which this synonym applies.
UK PROTEIN_CLASS_SYNONYM VARCHAR2(1000) The synonym for the protein class.
UK SYN_TYPE VARCHAR2(20) The type or origin of the synonym (e.g., ChEMBL, Concept Wiki, UMLS).

PROTEIN_CLASSIFICATION:
Table storing the protein family classifications for protein targets in ChEMBL (formerly in the target_class table)

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK PROTEIN_CLASS_ID NUMBER NOT NULL Primary key. Unique identifier for each protein family classification.
PARENT_ID NUMBER Protein_class_id for the parent of this protein family.
PREF_NAME VARCHAR2(500) Preferred/full name for this protein family.
SHORT_NAME VARCHAR2(50) Short/abbreviated name for this protein family (not necessarily unique).
PROTEIN_CLASS_DESC VARCHAR2(410) NOT NULL Concatenated description of each classification for searching purposes etc.
DEFINITION VARCHAR2(4000) Definition of the protein family.
CLASS_LEVEL NUMBER NOT NULL Level of the class within the hierarchy (level 1 = top level classification)

RELATIONSHIP_TYPE:
Lookup table for assays.relationship_type column, showing whether assays are mapped to targets of the correct identity and species ('Direct') or close homologues ('Homologue')

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK RELATIONSHIP_TYPE VARCHAR2(1) NOT NULL Relationship_type flag used in the assays table
RELATIONSHIP_DESC VARCHAR2(250) Description of relationship_type flags
SITE_COMPONENTS:
Table defining the location of the binding sites in the binding_sites table. A binding site could be defined in terms of which protein subunits (components) are involved, the domains within those subunits to which the compound binds, and possibly even the precise residues involved. For a target where the binding site is at the interface of two protein subunits or two domains, there will be two site_components describing each of these subunits/domains.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK SITECOMP_ID NUMBER NOT NULL Primary key.
FK,UK SITE_ID NUMBER NOT NULL Foreign key to binding_sites table.
FK,UK COMPONENT_ID NUMBER Foreign key to the component_sequences table, indicating which molecular component of the target is involved in the binding site.
FK,UK DOMAIN_ID NUMBER Foreign key to the domains table, indicating which domain of the given molecular component is involved in the binding site (where not known, the domain_id may be null).
SITE_RESIDUES VARCHAR2(2000) List of residues from the given molecular component that make up the binding site (where not know, will be null).

TARGET_COMPONENTS:
Links molecular target from the target_dictionary to the components they consist of (in the component_sequences table). For a protein complex or protein family target, for example, there will be multiple protein components in the component_sequences table.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
FK,UK TID NUMBER NOT NULL Foreign key to the target_dictionary, indicating the target to which the components belong.
FK,UK COMPONENT_ID NUMBER NOT NULL Foreign key to the component_sequences table, indicating which components belong to the target.
PK TARGCOMP_ID NUMBER NOT NULL Primary key.
HOMOLOGUE NUMBER NOT NULL Indicates that the given component is a homologue of the correct component (e.g., from a different species) when set to 1. This may be the case if the sequence for the correct protein/nucleic acid cannot be found in sequence databases. A value of 2 indicates that the sequence given is a representative of a species group, e.g., an E. coli protein to represent the target of a broad-spectrum antibiotic.

TARGET_DICTIONARY:
Target Dictionary containing all curated targets for ChEMBL. Includes both protein targets and non-protein targets (e.g., organisms, tissues, cell lines)

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK TID NUMBER NOT NULL Unique ID for the target
FK TARGET_TYPE VARCHAR2(30) Describes whether target is a protein, an organism, a tissue etc. Foreign key to TARGET_TYPE table.
PREF_NAME VARCHAR2(200) NOT NULL Preferred target name: manually curated
TAX_ID NUMBER NCBI taxonomy id of target
ORGANISM VARCHAR2(150) Source organism of molecuar target or tissue, or the target organism if compound activity is reported in an organism rather than a protein or tissue
FK,UK CHEMBL_ID VARCHAR2(20) NOT NULL ChEMBL identifier for this target (for use on web interface etc)
SPECIES_GROUP_FLAG NUMBER NOT NULL Flag to indicate whether the target represents a group of species, rather than an individual species (e.g., 'Bacterial DHFR'). Where set to 1, indicates that any associated target components will be a representative, rather than a comprehensive set.

TARGET_RELATIONS:
Table showing relationships between different protein targets based on overlapping protein components (e.g., relationship between a protein complex and the individual subunits).

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
FK TID NUMBER NOT NULL Identifier for target of interest (foreign key to target_dictionary table)
RELATIONSHIP VARCHAR2(20) NOT NULL Relationship between two targets (e.g., SUBSET OF, SUPERSET OF, OVERLAPS WITH)
FK RELATED_TID NUMBER NOT NULL Identifier for the target that is related to the target of interest (foreign key to target_dicitionary table)
PK TARGREL_ID NUMBER NOT NULL Primary key

TARGET_TYPE:
Lookup table for target types used in the target dictionary

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK TARGET_TYPE VARCHAR2(30) NOT NULL Target type (as used in target dictionary)
TARGET_DESC VARCHAR2(250) Description of target type
PARENT_TYPE VARCHAR2(25) Higher level classification of target_type, allowing grouping of e.g., all 'PROTEIN' targets, all 'NON-MOLECULAR' targets etc.

TISSUE_DICTIONARY:
Table storing information about tissues used in assays.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK TISSUE_ID NUMBER NOT NULL Primary key, numeric ID for each tissue.
UK UBERON_ID VARCHAR2(15) Uberon ontology identifier for this tissue.
UK PREF_NAME VARCHAR2(200) NOT NULL Name for the tissue (in most cases Uberon name).
UK EFO_ID VARCHAR2(20) Experimental Factor Ontology identifier for the tissue.
FK,UK CHEMBL_ID VARCHAR2(20) NOT NULL ChEMBL identifier for this tissue (for use on web interface etc)
BTO_ID VARCHAR2(20) BRENDA Tissue Ontology identifier for the tissue.
CALOHA_ID VARCHAR2(7) Swiss Institute for Bioinformatics CALOHA Ontology identifier for the tissue.
VARIANT_SEQUENCES:
Table storing information about mutant sequences and other variants used in assays. The sequence provided is a representative sequence incorporating the reported mutation/variant used in the assay - it is not necessarily the exact sequence used in the experiment.

KEYS COLUMN_NAME DATA_TYPE NULLABLE COMMENT
PK VARIANT_ID NUMBER NOT NULL Primary key, numeric ID for each sequence variant; -1 for unclassified variants.
UK MUTATION VARCHAR2(2000) Details of variant(s) used, with residue positions adjusted to match provided sequence.
UK ACCESSION VARCHAR2(25) UniProt accesion for the representative sequence used as the base sequence (without variation).
VERSION NUMBER Version of the UniProt sequence used as the base sequence.
ISOFORM NUMBER Details of the UniProt isoform used as the base sequence where relevant.
SEQUENCE CLOB Variant sequence formed by adjusting the UniProt base sequence with the specified mutations/variations.
ORGANISM VARCHAR2(200) Organism from which the sequence was obtained.
TAX_ID NUMBER NCBI Tax ID for the organism from which the sequence was obtained

1. ACTIVITIES Table

   Columns of Interest: ACTIVITY_ID, ASSAY_ID, STANDARD_TYPE, STANDARD_VALUE, STANDARD_UNITS, PCHEMBL_VALUE, BAO_ENDPOINT
   Reason: This table contains detailed information about the biological activities of compounds, including standardized activity types (e.g., IC50, EC50) and values. These data can provide insights into the efficacy and potency of different molecules, which is crucial for training a GNN to predict activity and selectivity profiles.

2. MOLECULE_DICTIONARY Table

   Columns of Interest: MOLREGNO, PREF_NAME, MOLECULE_TYPE, FIRST_APPROVAL, ORAL, TOPICAL, BLACK_BOX_WARNING, NATURAL_PRODUCT, PRODRUG
   Reason: This table stores information on compounds, including their types (e.g., small molecules, proteins), approval status, and any associated safety warnings. Understanding the types of molecules and their safety profiles will help the GNN make predictions that are not only effective but also safe.

3. COMPOUND_PROPERTIES Table

   Columns of Interest: MOLREGNO, MW_FREEBASE, ALOGP, HBA, HBD, PSA, AROMATIC_RINGS, HEAVY_ATOMS, QED_WEIGHTED
   Reason: Physicochemical properties such as molecular weight, lipophilicity (ALOGP), and hydrogen bond donor/acceptor counts are critical for training models that need to predict drug-like characteristics. These properties directly influence a molecule's absorption, distribution, metabolism, excretion, and toxicity (ADMET) profiles.

4. DRUG_MECHANISM Table

   Columns of Interest: MEC_ID, MECHANISM_OF_ACTION, TID, SITE_ID, ACTION_TYPE, DISEASE_EFFICACY
   Reason: This table provides details about the mechanism of action of drugs, which is crucial for understanding how molecules interact with biological targets. The data can help train the GNN to predict whether a new compound might activate or inhibit specific targets related to telomerase or other pathways of interest.

5. ASSAYS Table

   Columns of Interest: ASSAY_ID, DESCRIPTION, ASSAY_TYPE, ASSAY_TEST_TYPE, ASSAY_CATEGORY, ASSAY_ORGANISM, ASSAY_CELL_TYPE
   Reason: Assays provide experimental contexts (e.g., in vivo, in vitro) that can be used to predict how well a molecule might perform in similar biological settings. This information is crucial for predicting tissue-specific activities, aligning well with your target of selective telomerase activation in specific tissues.

6. TARGET_DICTIONARY Table

   Columns of Interest: TID, PREF_NAME, TARGET_TYPE, ORGANISM, TAX_ID
   Reason: Understanding the targets associated with each compound is critical for designing molecules that act on specific proteins or pathways. For telomerase activation, focusing on targets associated with the telomerase enzyme and related pathways will be key.

7. BINDING_SITES Table

   Columns of Interest: SITE_ID, SITE_NAME, TID
   Reason: Details on binding sites can provide information on where a molecule interacts with its target. This could be particularly useful for training the GNN on the specificity of binding, which is relevant to avoiding off-target effects in somatic cells.

8. LIGAND_EFF Table

   Columns of Interest: ACTIVITY_ID, BEI, SEI, LE, LLE
   Reason: Binding efficiency indices like BEI (Binding Efficiency Index) and SEI (Surface Efficiency Index) offer insights into the efficiency of molecule-target interactions, which is essential for optimizing compounds for higher selectivity and efficacy.

9. MOLECULE_ATC_CLASSIFICATION Table

   Columns of Interest: MOL_ATC_ID, LEVEL5, MOLREGNO
   Reason: Understanding the therapeutic classifications of compounds can help the GNN learn relationships between structure and therapeutic use, which may assist in predicting the therapeutic potential of novel compounds.

THIS IS THE MOLECUL DICTIONARY!!! I GAVE YOU THE SCHEMA ALREADY Table:
EXTRACT ALL OF THESE DATA FIELD ENTRIES
molecule_dictionary
Columns:
molregno bigint PK
pref_name varchar(255)
chembl_id varchar(20)
max_phase decimal(2,1)
therapeutic_flag tinyint
dosed_ingredient tinyint
structure_type varchar(10)
chebi_par_id bigint
molecule_type varchar(30)
first_approval int
oral tinyint
parenteral tinyint
topical tinyint
black_box_warning tinyint
natural_product tinyint
first_in_class tinyint
chirality tinyint
prodrug tinyint
inorganic_flag tinyint
usan_year int
availability_type tinyint
usan_stem varchar(50)
polymer_flag tinyint
usan_substem varchar(50)
usan_stem_definition varchar(1000)
indication_class varchar(1000)
withdrawn_flag tinyint
chemical_probe tinyint
orphan tinyint
Table: chembl_id_lookup
Columns:
chembl_id varchar(20) PK
entity_type varchar(50)
entity_id bigint
status varchar(10)
last_active mediumint

Table: cell_dictionary
Columns:
cell_id bigint PK
cell_name varchar(50)
cell_description varchar(200)
cell_source_tissue varchar(50)
cell_source_organism varchar(150)
cell_source_tax_id bigint
clo_id varchar(11)
efo_id varchar(12)
cellosaurus_id varchar(15)
cl_lincs_id varchar(8)
chembl_id varchar(20)
cell_ontology_id varchar(10)

Table: biotherapeutics
Columns:
molregno bigint PK
description varchar(2000)
helm_notation varchar(4000)

Table: binding_sites
Columns:
site_id bigint PK
site_name varchar(200)
tid bigint

Table: metabolism
Columns:
met_id bigint PK
drug_record_id bigint
substrate_record_id bigint
metabolite_record_id bigint
pathway_id bigint
pathway_key varchar(50)
enzyme_name varchar(200)
enzyme_tid bigint
met_conversion varchar(200)
organism varchar(100)
tax_id bigint
met_comment varchar(1000)

Table: molecule_synonyms
Columns:
molregno bigint
syn_type varchar(50)
molsyn_id bigint PK
res_stem_id bigint
synonyms varchar(250)

Table: predicted_binding_domains
Columns:
predbind_id bigint PK
activity_id bigint
site_id bigint
prediction_method varchar(50)
confidence varchar(10)

vTable: protein_class_synonyms
Columns:
protclasssyn_id bigint PK
protein_class_id bigint
protein_class_synonym varchar(1000)
syn_type varchar(20)

Table: protein_classification
Columns:
protein_class_id bigint PK
parent_id bigint
pref_name varchar(500)
short_name varchar(50)
protein_class_desc varchar(410)
definition varchar(4000)
class_level bigint

Table: target_components
Columns:
tid bigint
component_id bigint
targcomp_id bigint PK
homologue tinyint

Table: tissue_dictionary
Columns:
tissue_id bigint PK
uberon_id varchar(15)
pref_name varchar(200)
efo_id varchar(20)
chembl_id varchar(20)
bto_id varchar(20)
caloha_id varchar(7)

Table: variant_sequences
Columns:
variant_id bigint PK
mutation varchar(2000)
accession varchar(25)
version bigint
isoform bigint
sequence longtext
organism varchar(200)
tax_id bigint
