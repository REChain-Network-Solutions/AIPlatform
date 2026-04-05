<?php

use App\Http\Controllers\CustomModuleController;

Route::middleware(['auth', 'superadmin'])->group(function () {
    Route::resource('custom-modules', CustomModuleController::class);

    Route::post('custom-modules/analyze', [CustomModuleController::class, 'analyze'])
        ->name('custom-modules.analyze');

    Route::post('custom-modules/safe-fix', [CustomModuleController::class, 'safeFix'])
        ->name('custom-modules.safe-fix');

    Route::post('custom-modules/export-json', [CustomModuleController::class, 'exportJson'])
        ->name('custom-modules.export-json');

    Route::post('custom-modules/analyze-download', [CustomModuleController::class, 'downloadAnalysis'])
        ->name('custom-modules.analyze_download');

    // Rollback route for failed installations
    Route::post('custom-modules/{install}/rollback', [CustomModuleController::class, 'rollback'])
        ->name('custom-modules.rollback');
});



Route::prefix('admin/modules')->middleware(['auth', 'superadmin'])->group(function () {
    Route::get('/dashboard', [\App\Http\Controllers\ModuleInstallerDashboardController::class, 'index'])->name('modules.dashboard');
    Route::get('/history', [\App\Http\Controllers\ModuleInstallerDashboardController::class, 'history'])->name('modules.history');
    Route::get('/logs', [\App\Http\Controllers\ModuleInstallerDashboardController::class, 'logs'])->name('modules.logs');
    Route::get('/performance', [\App\Http\Controllers\ModuleInstallerDashboardController::class, 'performance'])->name('modules.performance');
    Route::post('/logs/export', [\App\Http\Controllers\ModuleInstallerDashboardController::class, 'exportLogs'])->name('modules.logs.export');
    Route::get('/{install}', [\App\Http\Controllers\ModuleInstallerDashboardController::class, 'show'])->name('modules.show');
});

