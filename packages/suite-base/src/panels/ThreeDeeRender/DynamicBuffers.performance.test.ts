// SPDX-FileCopyrightText: Copyright (C) 2023-2026 Bayerische Motoren Werke Aktiengesellschaft (BMW AG)<lichtblick@bmwgroup.com>
// SPDX-License-Identifier: MPL-2.0

import * as THREE from "three";

import { DynamicBufferGeometry } from "./DynamicBufferGeometry";
import { DynamicInstancedMesh } from "./DynamicInstancedMesh";

const scale = { x: 1, y: 2, z: 3 };
const color = { r: 1, g: 0.5, b: 0, a: 1 };
const points = (count: number) => Array.from({ length: count }, (_, x) => ({ x, y: 2, z: 3 }));

describe("dynamic rendering buffer allocation", () => {
  it("amortizes incremental vertex growth while preserving the exact draw count", () => {
    const geometry = new DynamicBufferGeometry();
    geometry.createAttribute("position", Float32Array, 3);
    let previous = geometry.attributes.position;
    let allocations = 0;
    for (let count = 1; count <= 1000; count++) {
      geometry.resize(count);
      if (geometry.attributes.position !== previous) {
        allocations++;
        previous = geometry.attributes.position;
      }
      expect(geometry.drawRange.count).toBe(count);
      expect(geometry.attributes.position!.count).toBeGreaterThanOrEqual(count);
    }
    expect(allocations).toBeLessThanOrEqual(20);
    geometry.dispose();
  });

  it("reuses exact-fit/shrinking vertex buffers and disposes old GPU attributes before replacement", () => {
    const geometry = new DynamicBufferGeometry();
    geometry.createAttribute("color", Uint8ClampedArray, 3, true);
    geometry.resize(100);
    const old = geometry.attributes.color;
    geometry.resize(100);
    geometry.resize(5);
    expect(geometry.attributes.color).toBe(old);
    const disposedAttributes: unknown[] = [];
    geometry.addEventListener("dispose", () => disposedAttributes.push(geometry.attributes.color));
    geometry.resize(101);
    expect(disposedAttributes).toEqual([old]);
    expect(geometry.attributes.color!.count).toBe(150);
    expect(geometry.attributes.color!.normalized).toBe(true);
    expect(geometry.attributes.color!.usage).toBe(THREE.DynamicDrawUsage);
    geometry.dispose();
  });

  it("does not partially replace attributes when an unsupported attribute is encountered", () => {
    const geometry = new DynamicBufferGeometry();
    geometry.createAttribute("position", Float32Array, 3);
    geometry.resize(10);
    const old = geometry.attributes.position;
    geometry.setAttribute("unsupported", new THREE.BufferAttribute(new Float32Array(10), 1));
    expect(() => geometry.resize(20)).toThrow("createAttribute");
    expect(geometry.attributes.position).toBe(old);
    geometry.dispose();
  });

  it("does not expand an exact-fit instanced mesh", () => {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.MeshBasicMaterial();
    const mesh = new DynamicInstancedMesh(geometry, material, 4);
    const old = mesh.instanceMatrix;
    mesh.set(points(4), scale, [], color);
    expect(mesh.instanceMatrix).toBe(old);
    expect(mesh.count).toBe(4);
    mesh.dispose();
    geometry.dispose();
    material.dispose();
  });

  it("allocates once for a large instance jump and never disposes shared geometry/material", () => {
    const geometry = new THREE.BufferGeometry();
    const material = new THREE.MeshBasicMaterial();
    const mesh = new DynamicInstancedMesh(geometry, material, 4);
    const old = mesh.instanceMatrix;
    const released: unknown[] = [];
    mesh.addEventListener("dispose", () => released.push(mesh.instanceMatrix));
    const geometryDispose = jest.spyOn(geometry, "dispose");
    const materialDispose = jest.spyOn(material, "dispose");
    mesh.set(points(10_000), scale, [], color);
    expect(released).toEqual([old]);
    expect(geometryDispose).not.toHaveBeenCalled();
    expect(materialDispose).not.toHaveBeenCalled();
    expect(mesh.instanceMatrix.array[9999 * 16 + 12]).toBe(9999);
    expect(mesh.instanceColor!.array[1]).toBe(127);
    const grown = mesh.instanceMatrix;
    mesh.set([], scale, [], color);
    expect(mesh.instanceMatrix).toBe(grown);
    expect(mesh.count).toBe(0);
    mesh.dispose();
    geometry.dispose();
    material.dispose();
  });
});
