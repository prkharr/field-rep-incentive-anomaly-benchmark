"""Tests for fallback data, canonical schema adaptation, and rep assignment."""

from __future__ import annotations

import pandas as pd

from field_rep_anomaly.data_loader import adapt_schema, generate_fallback_transactions
from field_rep_anomaly.synthetic_enrichment import create_rep_mapping


def test_fallback_generation_is_seeded_and_repeatable():
    first = generate_fallback_transactions(rows=180, start="2023-07-01", months=4, seed=123)
    second = generate_fallback_transactions(rows=180, start="2023-07-01", months=4, seed=123)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 180
    assert {"Customer", "Product Name", "Quantity", "Price", "Sales", "Month", "Year"} <= set(first)
    assert first[["Customer", "Product Name", "Month", "Year"]].notna().all().all()
    assert (first[["Quantity", "Price", "Sales"]].to_numpy() > 0).all()


def test_schema_adaptation_is_deterministic_and_derives_sales_from_aliases():
    source = pd.DataFrame(
        {
            "Account Name": ["A", "B"],
            "Brand": ["Cardiovex", "Respira"],
            "Qty": [2, 3],
            "Unit Price": [10.0, 4.0],
            "Period": ["Jan", "Feb"],
            "Sales Year": [2025, 2025],
            "Town": ["Pune", "Pune"],
            "Nation": ["India", "India"],
        }
    )

    first, first_audit = adapt_schema(source)
    second, second_audit = adapt_schema(source)

    pd.testing.assert_frame_equal(first, second)
    assert first_audit == second_audit
    assert source.columns.tolist()[0] == "Account Name"  # adaptation does not mutate its input
    assert first["customer"].tolist() == ["A", "B"]
    assert first["product_name"].tolist() == ["Cardiovex", "Respira"]
    assert first["sales"].tolist() == [20.0, 12.0]
    assert first["date"].tolist() == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")]
    assert first_audit["canonical_mapping"]["sales"] == "DERIVED: quantity * price"
    assert "sales" in first_audit["derived_or_defaulted_fields"]


def test_rep_mapping_is_order_independent_and_one_rep_per_customer_org(canonical_transactions):
    assigned, mapping = create_rep_mapping(canonical_transactions, reps_per_territory=3, seed=29)
    assigned_again, mapping_again = create_rep_mapping(canonical_transactions, reps_per_territory=3, seed=29)
    shuffled = canonical_transactions.sample(frac=1.0, random_state=91).reset_index(drop=True)
    _, shuffled_mapping = create_rep_mapping(shuffled, reps_per_territory=3, seed=29)

    pd.testing.assert_frame_equal(assigned, assigned_again)
    pd.testing.assert_frame_equal(mapping, mapping_again)
    pd.testing.assert_frame_equal(mapping, shuffled_mapping)

    org_customer = ["sales_team", "sales_manager", "territory_id", "customer"]
    assert not mapping.duplicated(org_customer).any()
    assert mapping.groupby(org_customer, observed=True)["rep_id"].nunique().eq(1).all()
    assert assigned.groupby(org_customer, observed=True)["rep_id"].nunique().eq(1).all()
    assert assigned["rep_id"].notna().all()
    assert mapping["rep_slot"].between(0, 2).all()
