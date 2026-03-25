<?php

use Illuminate\Support\Facades\Route;

Route::middleware(['api', 'auth:sanctum'])->prefix('api/budgetallocationaprovalmodule')->group(function() {
    Route::get('/', function() {
        return response()->json(['module' => 'budgetallocationaprovalmodule', 'ok' => true]);
    });
});
