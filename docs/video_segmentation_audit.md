# Video Segmentation Audit

## Scope

This audit compares the four unreviewed raw videos:

- `cam_44`
- `cam_46`
- `cam_33`
- `cam_3`

The goal is to decide which video is best for Stage 2 segmentation and method-study style labeling.

## Evaluation Criteria

- workstation visibility
- worker-to-machine clarity
- hand/fabric interaction readability
- occlusion from neighboring workers
- fabric pile interference
- stability for defining repeatable station ROIs later

## Recommendation

Best overall segmentation video: `cam_33`

Why:

- sewing machine zones are clearer than the other candidates
- the main worker actions are easier to separate from neighboring stations
- the top-center and lower-center work areas remain readable across multiple samples
- less white-fabric washout than `cam_44`
- less full-frame clutter than `cam_3`

Second-best fallback: `cam_46`

Why:

- machine regions are visible
- dark fabric gives better hand/fabric contrast
- but neighboring workers and cloth spread still create frequent cross-station interference

Not recommended as primary segmentation source:

- `cam_44`: good worker visibility, but heavy white-fabric spread and frequent crowd overlap make segment boundaries noisy
- `cam_3`: too cluttered, with dense fabric piles covering too much of the table and machine areas

## Candidate Summary

See [video_station_comparison_sheet.csv](/media/milab-1/009a6625-83db-44d1-8d42-364400c9fc34/Mritunjoys' Workplace/RMG/artifacts/video_audit/video_station_comparison_sheet.csv)

## Contact Sheets

- [cam_44_contact_sheet.jpg](/media/milab-1/009a6625-83db-44d1-8d42-364400c9fc34/Mritunjoys' Workplace/RMG/artifacts/video_audit/cam_44_contact_sheet.jpg)
- [cam_46_contact_sheet.jpg](/media/milab-1/009a6625-83db-44d1-8d42-364400c9fc34/Mritunjoys' Workplace/RMG/artifacts/video_audit/cam_46_contact_sheet.jpg)
- [cam_33_contact_sheet.jpg](/media/milab-1/009a6625-83db-44d1-8d42-364400c9fc34/Mritunjoys' Workplace/RMG/artifacts/video_audit/cam_33_contact_sheet.jpg)
- [cam_3_contact_sheet.jpg](/media/milab-1/009a6625-83db-44d1-8d42-364400c9fc34/Mritunjoys' Workplace/RMG/artifacts/video_audit/cam_3_contact_sheet.jpg)
