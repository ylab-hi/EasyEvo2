import pyfastx


def test_build_variant_records_snv(tmp_path):
    """Test building ref/alt sequences from a simple SNV VCF."""
    # Create a minimal reference FASTA
    ref_path = tmp_path / "ref.fa"
    # 100bp reference sequence
    seq = "A" * 50 + "C" + "G" * 49
    ref_path.write_text(f">chr1\n{seq}\n")

    # Create a minimal VCF with one SNV
    vcf_path = tmp_path / "test.vcf"
    vcf_content = (
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t51\t.\tC\tT\t.\t.\t.\n"
    )
    vcf_path.write_text(vcf_content)

    from easyevo2.vep import _build_variant_records

    ref_fasta = pyfastx.Fasta(str(ref_path))
    records, unique_ref_seqs, _ = _build_variant_records(
        vcf_path, ref_fasta, context_length=100
    )

    assert len(records) == 1
    assert records[0]["chrom"] == "chr1"
    assert records[0]["pos"] == 51
    assert records[0]["ref"] == "C"
    assert records[0]["alt"] == "T"
    # Alt sequence should have T at the variant position
    alt_seq = records[0]["alt_seq"]
    assert "T" in alt_seq
    # Reference should have C at that position
    assert "C" in unique_ref_seqs[0]
