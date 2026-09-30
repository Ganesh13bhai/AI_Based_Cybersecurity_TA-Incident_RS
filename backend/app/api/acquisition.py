from fastapi import APIRouter, HTTPException

from app.services.data_acquisition import DataAcquisitionService


router = APIRouter(
    prefix="/api/v1/acquisition",
    tags=["Data Acquisition"]
)


@router.get("/datasets")
def get_datasets():
    """
    Discover registered cybersecurity datasets
    and their available files.
    """

    service = DataAcquisitionService()

    try:
        datasets = service.scan_datasets()

        return {
            "status": "success",
            "dataset_count": len(datasets),
            "datasets": datasets
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )