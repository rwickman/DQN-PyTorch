import random

class ReplayMemory:
    def __init__(self, args):
        self._capacity = args.replay_capacity
        self._memory = []
        # Pointer to end of memory
        self._cur_pos = 0
        self._batch_size = args.batch_size
    
    def append(self, e_t):
        """Append experience."""
        if len(self._memory) >= self._capacity:
            self._memory[self._cur_pos] = e_t
        else:
            self._memory[self._cur_pos] = e_t
        
        # Update end of memory
        self._cur_pos = (self._cur_pos + 1) %  self._capacity 

    def sample(self):
        """Sample batch size experience replay."""
        return random.sample(self._memory, self._batch_size)