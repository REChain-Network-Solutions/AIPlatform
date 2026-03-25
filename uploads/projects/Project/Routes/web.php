<?php

// use App\Http\Controllers\Modules;
// use Illuminate\Support\Facades\Route;

Route::middleware('web', 'authh', 'SetSessionData', 'auth', 'language', 'timezone', 'AdminSidebarMenu')->prefix('project')->group(function () {
    Route::put('project/{id}/post-status', [Modules\Project\Http\Controllers\ProjectController::class, 'postProjectStatus']);
    Route::put('project-settings', [Modules\Project\Http\Controllers\ProjectController::class, 'postSettings']);
    Route::resource('project', 'Modules\Project\Http\Controllers\ProjectController');
    Route::resource('project-task', 'Modules\Project\Http\Controllers\TaskController');
    Route::get('project-task-get-status', [Modules\Project\Http\Controllers\TaskController::class, 'getTaskStatus']);
    Route::put('project-task/{id}/post-status', [Modules\Project\Http\Controllers\TaskController::class, 'postTaskStatus']);
    Route::put('project-task/{id}/post-description', [Modules\Project\Http\Controllers\TaskController::class, 'postTaskDescription']);
    Route::resource('project-task-comment', 'Modules\Project\Http\Controllers\TaskCommentController');
    Route::post('post-media-dropzone-upload', [Modules\Project\Http\Controllers\TaskCommentController::class, 'postMedia']);
    Route::resource('project-task-time-logs', 'Modules\Project\Http\Controllers\ProjectTimeLogController');
    Route::resource('activities', 'Modules\Project\Http\Controllers\ActivityController')->only(['index']);
    Route::get('project-invoice-tax-report', [Modules\Project\Http\Controllers\InvoiceController::class, 'getProjectInvoiceTaxReport']);
    Route::resource('invoice', 'Modules\Project\Http\Controllers\InvoiceController');
    Route::get('project-employee-timelog-reports', [Modules\Project\Http\Controllers\ReportController::class, 'getEmployeeTimeLogReport']);
    Route::get('project-timelog-reports', [Modules\Project\Http\Controllers\ReportController::class, 'getProjectTimeLogReport']);
    Route::get('project-reports', [Modules\Project\Http\Controllers\ReportController::class, 'index']);

    Route::get('/install', [Modules\Project\Http\Controllers\InstallController::class, 'index']);
    Route::post('/install', [Modules\Project\Http\Controllers\InstallController::class, 'install']);
    Route::get('/install/uninstall', [Modules\Project\Http\Controllers\InstallController::class, 'uninstall']);
    Route::get('/install/update', [Modules\Project\Http\Controllers\InstallController::class, 'update']);

    Route::get('settings', [Modules\Project\Http\Controllers\SettingsController::class, 'index']);
    Route::post('settings', [Modules\Project\Http\Controllers\SettingsController::class, 'store']);

});
