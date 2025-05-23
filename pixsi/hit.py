class Hit:
    def __init__(self,hit_ID , tpc_ID , event_ID, pixel_ID, type, charge, start_time, end_time):
        """
        Initialize a Hit object. Charge is uniform on a given interval

        Parameters:
        - charge (float): The charge of the hit.
        - start_time (float): The start time of the hit.
        - delta_time (float): The duration of the hit.
        """
        self.tpc_ID = tpc_ID
        self.event_ID = event_ID
        self.hit_ID = hit_ID
        self.pixel_ID = pixel_ID
        self.charge = charge
        self.start_time = start_time
        self.end_time = end_time
        self.type=type

    def __repr__(self):
        """
        Return a string representation of the Hit object.
        """
        return f"Hit(charge={self.charge}, start_time={self.start_time}, end_time={self.end_time})"


