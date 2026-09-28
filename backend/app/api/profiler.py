from fastapi import APIRouter, HTTPException

from app.services.data_profiler import DataProfilerService


router = APIRouter(
    prefix="/api/v1/profile",
    tags=["Data Profiler"]
)


@router.get("/datasets")
def profile_datasets():

    service = DataProfilerService()

    try:

        result = service.profile_datasets()

        return result

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
