<?php

use App\Extensions\AIImagePro\System\Http\Controllers\AIImageProController;

beforeEach(function () {
    $this->controller = new AIImageProController;
    $this->method = new ReflectionMethod(AIImageProController::class, 'parseDimensionsFromAspectRatio');
    $this->method->setAccessible(true);
});

test('parses pixel dimensions format correctly', function () {
    $result = $this->method->invoke($this->controller, '1024x1024');
    expect($result)->toBe(['width' => 1024, 'height' => 1024]);

    $result = $this->method->invoke($this->controller, '1792x1024');
    expect($result)->toBe(['width' => 1792, 'height' => 1024]);

    $result = $this->method->invoke($this->controller, '1024x1792');
    expect($result)->toBe(['width' => 1024, 'height' => 1792]);
});

test('parses ratio format correctly', function () {
    $result = $this->method->invoke($this->controller, '1:1');
    expect($result['width'])->toBe(1024);
    expect($result['height'])->toBe(1024);

    $result = $this->method->invoke($this->controller, '16:9');
    expect($result['width'])->toBe(1024);
    expect($result['height'])->toBe(576);

    $result = $this->method->invoke($this->controller, '9:16');
    expect($result['width'])->toBe(576);
    expect($result['height'])->toBe(1024);

    $result = $this->method->invoke($this->controller, '4:3');
    expect($result['width'])->toBe(1024);
    expect($result['height'])->toBe(768);

    $result = $this->method->invoke($this->controller, '3:4');
    expect($result['width'])->toBe(768);
    expect($result['height'])->toBe(1024);
});

test('parses named ratios correctly', function () {
    $result = $this->method->invoke($this->controller, 'square');
    expect($result['width'])->toBe(1024);
    expect($result['height'])->toBe(1024);

    $result = $this->method->invoke($this->controller, 'square_hd');
    expect($result['width'])->toBe(1024);
    expect($result['height'])->toBe(1024);

    $result = $this->method->invoke($this->controller, 'portrait_4_3');
    expect($result['width'])->toBe(768);
    expect($result['height'])->toBe(1024);

    $result = $this->method->invoke($this->controller, 'landscape_16_9');
    expect($result['width'])->toBe(1024);
    expect($result['height'])->toBe(576);

    $result = $this->method->invoke($this->controller, 'portrait_16_9');
    expect($result['width'])->toBe(576);
    expect($result['height'])->toBe(1024);
});

test('returns default dimensions for null or empty aspect ratio', function () {
    $result = $this->method->invoke($this->controller, null);
    expect($result)->toBe(['width' => 1024, 'height' => 1024]);

    $result = $this->method->invoke($this->controller, '');
    expect($result)->toBe(['width' => 1024, 'height' => 1024]);
});

test('returns default dimensions for unknown format', function () {
    $result = $this->method->invoke($this->controller, 'unknown_format');
    expect($result)->toBe(['width' => 1024, 'height' => 1024]);
});
