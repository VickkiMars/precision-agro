# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Commercial farm owners, agricultural field operators, and drone scout pilots managing crop monitoring directly from a laptop or rugged tablet at the field edge in demanding rural environments.

## Product Purpose

Deliver an autonomous, edge-computed precision agriculture system that combines interactive field boundary planning, real-time drone telemetry, and on-device computer vision disease detection to generate actionable spot-spraying chemical prescriptions without cloud dependence. Success is measured by reducing aggregate chemical application volume while eliminating crop pathology hotspots before disease spreads.

## Positioning

A zero-cloud, fully offline cyber-physical drone solution pairing an onboard deep learning vision engine and embedded agronomic treatment database with a lightweight, browser-based Ground Control Station. Designed specifically for low-bandwidth and harsh rural settings (including Harmattan dust conditions in Sub-Saharan Africa).

## Operating Context

- **Environment:** Outdoor agricultural field margins, bright direct sunlight, dusty conditions, intermittent or completely absent cellular connectivity.
- **Hardware Loop:** Raspberry Pi 4 Companion PC communicating with a Pixhawk 4 Autopilot via UART MAVLink (`/dev/ttyAMA0 @ 57600 baud`) and a Sony IMX219 CSI camera.
- **Operator Workflow:** Pre-flight farm boundary demarcation -> Flight pattern selection -> Pre-arm diagnostic confirmation -> Autonomous scouting survey -> Post-flight prescription export to CSV/GeoJSON for tractor sprayers.

## Capabilities and Constraints

- **Interactive Parcel Demarcation:** Dynamic polygon and box boundary drawing with geodesic area calculation in hectares and acres.
- **Flight Path Preset Engine:** 5 selectable flight geometries (Standard Serpentine, Row-Optimized / Long-Axis, Crosshatch 3D Orthogonal, Perimeter Scout Buffer, and Targeted Hotspot Orbit) with automated track spacing and optical swath sizing.
- **Zero-Dependency Architecture:** GCS server is implemented purely in Python standard library (`http.server` + `sqlite3`), enabling identical execution on edge companion hardware and stateless container runtimes (Vercel Docker).
- **Embedded Agronomic Matrix:** Offline SQLite database (`crop_health_edge.db` in WAL mode) linking 38 plant disease classes with recommended active ingredients, application rates (ml/L), and safety intervals.
- **Optical Environment Adaptation:** Integrated Harmattan dust haze compensation and exposure calibration for optical sensors.
- **Constraint:** Deep learning inference (PyTorch) and MAVLink serial communications run exclusively on the physical edge companion computer; the web GCS interface remains ultra-lightweight and decoupled from heavy machine learning runtimes.

## Brand Commitments

- **Name:** Precision Agriculture Drone Ground Control Station (Precision Agro GCS).
- **Visual Voice:** High-contrast field-edge aesthetic strictly disciplined around **White, Gray, and Green** (`#ffffff`, `#f8fafc`, `#f1f5f9`, `#e2e8f0`, `#15803d`), crisp typography (`Inter` + `JetBrains Mono`), and clear semantic status indicators without hard borders, loud badges, or visual clutter.
- **Color Discipline:** Luminous crisp white (`#ffffff`) for elevated cards, modals, and navigation controls; calm slate and cool grays (`#f8fafc` canvas, `#f1f5f9` containers, `#475569` secondary actions, `#0f172a` high-contrast typography); and rich agronomic greens (`#15803d` primary emerald, `#14532d` forest, `#dcfce7` tint container) for nominal flight status, field boundaries, waypoints, and active transects.

## Evidence on Hand

- Technical specification: `PRECISION AGRICULTURE WITH DRONE BASE SOLUTION TECHNOLOGY FULL PROJECT (1).docx`
- Deep learning classifier weights: `plant-disease-model.pth` (ResNet-9 architecture trained on 38 disease categories)
- Field telemetry and prescription logs: `field_prescription_log.csv` and `field_prescription_map.geojson`
- Optical configuration: `camera_config.json` with Harmattan dust adaptive mode
- Unit test suite: `test_system.py` verifying database, vision, telemetry, prescription, and mission planning benchmarks

## Product Principles

1. **Edge-First Autonomy:** Every critical capability—from waypoint planning to disease identification and chemical prescription—must function flawlessly in an isolated field without internet connectivity.
2. **Turnkey Operational Velocity:** A farmer at the edge of a field needs immediate actionable simplicity: draw the boundary, pick the preset, verify system health, and upload in under 60 seconds.
3. **Agronomic Ground Truth:** Prescriptions must reflect real, dosage-accurate agronomic guidelines with explicit safety intervals rather than opaque machine learning confidence scores.
4. **Field-Edge Legibility:** Controls, telemetry values, and map overlays must remain effortlessly readable under direct sunlight on tablet touchscreens.

## Accessibility & Inclusion

- **Sunlight Legibility:** High contrast text and borders designed to combat screen glare outdoors.
- **Dual-Coded Telemetry:** Status indicators pair semantic colors with explicit textual labels and iconography (never color alone).
- **Touch Ergonomics:** Generous click and tap targets (minimum 40px touch boundaries) suitable for outdoor tablet usage with work gloves.
