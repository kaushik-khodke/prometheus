# Problem Statement: [ST-02] Satellite-Based Hyperlocal Disaster Detection and Impact Mapping

**Domain:** Space Tech  
**Category:** Software / Artificial Intelligence / Earth Observation (EO)  
**Target Event / Problem Code:** ST-02  

---

## 1. Problem Statement Title & Overview

### **Title:**
Satellite-Based Hyperlocal Disaster Detection and Impact Mapping

### **Overview:**
Build automated deep learning computer vision pipelines that process multispectral satellite imagery to detect emerging disasters (floods, wildfires, landslides, urban damage) at hyperlocal resolution.

---

## 2. Core Operational Scope & Challenges

1. **Hyperlocal Precision (Sub-meter to 10m Ground Resolution)**:
   - Identify critical disaster boundaries at the individual parcel, structure, and riverbank level.
   - Accurately compute physical affected areas in square meters ($m^2$) and hectares ($ha$).

2. **Multi-Hazard Detection Vectors**:
   - **Floods & Inundations**: Rapid detection of water pooling, breached flood defenses, and submerged road networks using radiometric indices (NDWI / MNDWI) and grounding segmentation.
   - **Wildfires & Thermal Scars**: Narrowband hyperspectral absorption profiling, NBR (Normalized Burn Ratio), and Reed-Xiaoli (RX) anomaly detection for identifying active firefronts and burn severity.
   - **Landslides & Ground Displacements**: All-weather, cloud-penetrating synthetic aperture radar (SAR) backscatter fusion with optical imagery to detect slope shearing and debris flows through storm clouds.
   - **Urban Damage & Structural Collapse**: Bi-temporal Siamese difference transformers isolating building collapses and infrastructure wreckage post-disaster.

3. **Autonomous Agentic Orchestration**:
   - Automatically ingest multi-sensor imagery (optical, SAR, hyperspectral), route queries to specialized deep learning pipelines, and generate actionable evidence overlays without manual GIS parameterization.

4. **Observability & Operational Telemetry**:
   - Trace disaster triage decisions, model latencies, and execution telemetry through Langfuse Cloud for mission auditability.
