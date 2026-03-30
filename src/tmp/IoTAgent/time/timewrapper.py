import datetime
from datetime import datetime, timezone

class TimeWrapperFunctions:
    def currentTime(self) -> str:
        """Return the current time in ISO format
        Args: None
        Returns:
            the current time UTC as an ISO-formatted string
        """
        now = datetime.now(timezone.utc)
        
        nowISO = now.isoformat()

        return nowISO
