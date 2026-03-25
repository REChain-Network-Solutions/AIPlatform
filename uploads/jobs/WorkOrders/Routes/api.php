<?php

use Illuminate\Support\Facades\Route;

if (config('workorders.api_auth', true)) {
    Route::middleware(['auth','can:workorders.view'])->prefix('workorders')->group(function () {
        Route::get('/health', fn() => response()->json(['ok'=>true]));
        // TODO: add protected endpoints
    });
} else {
    Route::prefix('workorders')->group(function () {
        Route::get('/health', fn() => response()->json(['ok'=>true]));
    });
}
