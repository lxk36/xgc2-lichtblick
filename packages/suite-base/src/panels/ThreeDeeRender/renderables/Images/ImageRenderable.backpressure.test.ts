/** @jest-environment jsdom */

// SPDX-FileCopyrightText: Copyright (C) 2023-2026 Bayerische Motoren Werke Aktiengesellschaft (BMW AG)<lichtblick@bmwgroup.com>
// SPDX-License-Identifier: MPL-2.0

import { IRenderer } from "@lichtblick/suite-base/panels/ThreeDeeRender/IRenderer";

import {
  ImageRenderable,
  IMAGE_RENDERABLE_DEFAULT_SETTINGS,
  ImageUserData,
} from "./ImageRenderable";
import { AnyImage } from "./ImageTypes";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

const sample = (seq: number): AnyImage => ({
  format: "jpeg",
  data: new Uint8Array([seq]),
  header: { frame_id: "camera", stamp: { sec: seq, nsec: 0 } },
});

async function settle(): Promise<void> {
  for (let i = 0; i < 12; i++) {
    await Promise.resolve();
  }
}

function fixture() {
  const errors = {
    add: jest.fn(),
    addToTopic: jest.fn(),
    remove: jest.fn(),
    removeFromTopic: jest.fn(),
  };
  const renderer = { queueAnimationFrame: jest.fn(), settings: { errors } } as unknown as IRenderer;
  const data: ImageUserData = {
    topic: "camera",
    settings: { ...IMAGE_RENDERABLE_DEFAULT_SETTINGS },
    firstMessageTime: 0n,
    cameraInfo: undefined,
    cameraModel: undefined,
    image: undefined,
    texture: undefined,
    material: undefined,
    geometry: undefined,
    mesh: undefined,
    frameId: "camera",
    messageTime: 0n,
    receiveTime: 0n,
    pose: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    settingsPath: [],
  };
  const renderable = new ImageRenderable("camera", renderer, data);
  const update = jest.spyOn(renderable, "update").mockImplementation(() => {});
  const requests: ReturnType<typeof deferred<ImageBitmap | ImageData>>[] = [];
  const decode = jest
    .spyOn(
      renderable as unknown as {
        decodeImage: (image: AnyImage, width?: number) => Promise<ImageBitmap | ImageData>;
      },
      "decodeImage",
    )
    .mockImplementation(() => {
      const request = deferred<ImageBitmap | ImageData>();
      requests.push(request);
      return request.promise;
    });
  return { renderable, requests, decode, update, errors };
}

describe("independent image decode backpressure", () => {
  afterEach(() => jest.restoreAllMocks());

  it("bounds a 1000-frame burst to two active decodes and the latest waiting image", async () => {
    const { renderable, requests, decode, update } = fixture();
    const images = Array.from({ length: 1000 }, (_, i) => sample(i));
    images.forEach((image, i) => renderable.setImage(image, i));
    expect(decode).toHaveBeenCalledTimes(2);
    requests[0]!.resolve(new ImageBitmap());
    await settle();
    expect(decode).toHaveBeenCalledTimes(3);
    expect(decode).toHaveBeenLastCalledWith(images[999], 999);
    expect(update).toHaveBeenCalledTimes(1); // Active results must not starve under sustained load.
    requests[1]!.resolve(new ImageBitmap());
    requests[2]!.resolve(new ImageBitmap());
    await settle();
    expect(decode).toHaveBeenCalledTimes(3);
    renderable.dispose();
  });

  it("drops pending work and closes late bitmaps after disposal", async () => {
    const { renderable, requests, decode, update } = fixture();
    const first = new ImageBitmap();
    const second = new ImageBitmap();
    const closeFirst = jest.spyOn(first, "close");
    const closeSecond = jest.spyOn(second, "close");
    renderable.setImage(sample(1));
    renderable.setImage(sample(2));
    renderable.setImage(sample(3));
    renderable.dispose();
    renderable.setImage(sample(4));
    requests[0]!.resolve(first);
    requests[1]!.resolve(second);
    await settle();
    expect(decode).toHaveBeenCalledTimes(2);
    expect(closeFirst).toHaveBeenCalledTimes(1);
    expect(closeSecond).toHaveBeenCalledTimes(1);
    expect(update).not.toHaveBeenCalled();
  });

  it("invalidates pre-seek active and waiting images without wedging decode slots", async () => {
    const { renderable, requests, decode, update } = fixture();
    renderable.setImage(sample(1));
    renderable.setImage(sample(2));
    renderable.setImage(sample(3));
    renderable.resetVideoForSeek();
    requests[0]!.resolve(new ImageBitmap());
    requests[1]!.resolve(new ImageBitmap());
    await settle();
    expect(decode).toHaveBeenCalledTimes(2);
    expect(update).not.toHaveBeenCalled();
    renderable.setImage(sample(4));
    expect(decode).toHaveBeenCalledTimes(3);
    renderable.dispose();
    requests[2]!.resolve(new ImageBitmap());
    await settle();
  });

  it("does not report an older decode failure over a newer successfully displayed image", async () => {
    const { renderable, requests, errors } = fixture();
    renderable.setImage(sample(1));
    renderable.setImage(sample(2));
    const latest = new ImageBitmap();
    requests[1]!.resolve(latest);
    await settle();
    requests[0]!.reject(new Error("stale decode failure"));
    await settle();
    expect(renderable.getDecodedImage()).toBe(latest);
    expect(errors.add).not.toHaveBeenCalled();
    renderable.dispose();
  });
});
