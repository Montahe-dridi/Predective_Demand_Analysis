import pandas as pd

def transform_freight_cost(fact_shipments: pd.DataFrame, dim_freight: pd.DataFrame) -> pd.DataFrame:
    print("🔄 Transforming FreightCostAnalysis...")

    # Ensure keys match
    if "ShipmentID" in fact_shipments.columns and "ShipmentKey" not in fact_shipments.columns:
        fact_shipments = fact_shipments.rename(columns={"ShipmentID": "ShipmentKey"})

    # Merge Fact with DimFreightType
    df = fact_shipments.merge(
        dim_freight[["FreightTypeKey", "FreightTypeName"]],
        on="FreightTypeKey",
        how="left"
    )

    # Compute cost metrics
    df["CostVariance"] = df["ActualFreightCost"] - df["PlannedFreightCost"]
    df["VarianceFlag"] = df["CostVariance"].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
    df["CostEfficiencyRatio"] = df.apply(
        lambda row: row["ActualFreightCost"] / row["PlannedFreightCost"]
        if row["PlannedFreightCost"] not in [0, None, float("nan")] else None,
        axis=1
    )

    print("✅ FreightCostAnalysis transformation complete.")

    return df[
        [
            "ShipmentKey",
            "FreightTypeKey",
            "PlannedFreightCost",
            "ActualFreightCost",
            "CostVariance",
            "VarianceFlag",
            "CostEfficiencyRatio"
        ]
    ]
