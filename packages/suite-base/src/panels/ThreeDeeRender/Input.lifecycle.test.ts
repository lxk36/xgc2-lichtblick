/** @jest-environment jsdom */

// SPDX-FileCopyrightText: Copyright (C) 2023-2026 Bayerische Motoren Werke Aktiengesellschaft (BMW AG)<lichtblick@bmwgroup.com>
// SPDX-License-Identifier: MPL-2.0

import * as THREE from "three";

import { Input } from "./Input";

describe("3D input scheduling and teardown", () => {
  const originalObserver = globalThis.ResizeObserver;
  let notify: ResizeObserverCallback;
  let input: Input;
  let parent: HTMLDivElement;
  let frames: Map<number, FrameRequestCallback>;
  let disconnected: jest.Mock;

  beforeEach(() => {
    frames = new Map();
    disconnected = jest.fn();
    globalThis.ResizeObserver = class {
      public constructor(callback: ResizeObserverCallback) {
        notify = callback;
      }
      public observe(): void {}
      public unobserve(): void {}
      public disconnect = disconnected;
    };
    let nextId = 0;
    jest.spyOn(window, "requestAnimationFrame").mockImplementation((callback) => {
      const id = nextId++;
      frames.set(id, callback);
      return id;
    });
    jest.spyOn(window, "cancelAnimationFrame").mockImplementation((id) => {
      frames.delete(id);
    });
    parent = document.createElement("div");
    parent.style.cssText = "padding: 0px; border: 0px solid transparent";
    const canvas = document.createElement("canvas");
    parent.appendChild(canvas);
    document.body.appendChild(parent);
    Object.defineProperties(parent, {
      clientWidth: { configurable: true, value: 100 },
      clientHeight: { configurable: true, value: 80 },
    });
    input = new Input(canvas, () => new THREE.PerspectiveCamera());
  });

  afterEach(() => {
    input.dispose();
    parent.remove();
    globalThis.ResizeObserver = originalObserver;
    jest.restoreAllMocks();
  });

  const resized = () => notify([], {} as ResizeObserver);

  it("coalesces resize bursts into one display-frame update at the latest size", () => {
    const changed = jest.fn();
    input.on("resize", changed);
    for (let width = 200; width < 300; width++) {
      Object.defineProperty(parent, "clientWidth", { configurable: true, value: width });
      resized();
    }
    expect(frames.size).toBe(1);
    expect(changed).not.toHaveBeenCalled();
    frames.get(0)!(0);
    expect(changed).toHaveBeenCalledTimes(1);
    expect(input.canvasSize.width).toBe(299);
  });

  it("cancels frame id zero on disposal and ignores late observer delivery", () => {
    resized();
    input.dispose();
    expect(window.cancelAnimationFrame).toHaveBeenCalledWith(0);
    expect(frames.size).toBe(0);
    resized();
    expect(frames.size).toBe(0);
    expect(disconnected).toHaveBeenCalled();
  });

  it("replaces a prior drag and removes window listeners on blur/disposal", () => {
    const old = jest.fn();
    const current = jest.fn();
    const move = () =>
      window.dispatchEvent(new MouseEvent("mousemove", { clientX: 10, clientY: 10 }));
    input.trackDrag(old);
    input.trackDrag(current);
    move();
    expect(old).not.toHaveBeenCalled();
    expect(current).toHaveBeenCalledTimes(1);
    window.dispatchEvent(new Event("blur"));
    move();
    expect(current).toHaveBeenCalledTimes(1);
    input.trackDrag(current);
    input.dispose();
    move();
    input.trackDrag(old);
    move();
    expect(current).toHaveBeenCalledTimes(1);
    expect(old).not.toHaveBeenCalled();
  });
});
