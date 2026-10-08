// Face tracking off the main thread: a detection can take tens of milliseconds, and here it
// can't stall rendering. The tracking itself is in facetracker.js.

import { createFaceTracker } from './facetracker.js';

const handle = createFaceTracker((message) => self.postMessage(message));

self.onmessage = async ({ data }) => {
  await handle(data);
  if (data.type === 'close') self.close();
};
