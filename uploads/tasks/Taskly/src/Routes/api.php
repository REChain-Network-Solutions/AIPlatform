<?php

use Illuminate\Http\Request;
use Workdo\Taskly\Http\Controllers\Api\ProjectApiController;
use Workdo\Taskly\Http\Controllers\Api\ProjectDashboardApiController;
use Workdo\Taskly\Http\Controllers\Api\TaskApiController;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| is assigned the "api" middleware group. Enjoy building your API!
|
*/

Route::middleware('auth:api')->get('/taskly', function (Request $request) {
    return $request->user();
});


Route::prefix('Taskly')->group(function () {
    Route::middleware(['jwt.api.auth'])->group(function () {

        Route::post('home',[ProjectDashboardApiController::class,'index']);
        Route::post('get-workspace-users',[ProjectApiController::class,'getWorkspaceUsers']);

        Route::post('project-list',[ProjectApiController::class,'index']);
        Route::post('project-details',[ProjectApiController::class,'projectDetails']);
        Route::post('project-activity',[ProjectApiController::class,'projectActivity']);
        Route::post('project-create-update',[ProjectApiController::class,'projectCreateAndUpdate']);
        Route::post('project-delete',[ProjectApiController::class,'destroyProject']);
        Route::post('project-status-update',[ProjectApiController::class,'projectStatusUpdate']);

        Route::post('task-list',[TaskApiController::class,'index']);
        Route::post('taskboard',[TaskApiController::class,'taskboard']);
        Route::post('task-details',[TaskApiController::class,'taskDetails']);
        Route::post('task-create-update',[TaskApiController::class,'taskCreateAndUpdate']);
        Route::post('task-stage-update',[TaskApiController::class,'taskStageUpdate']);
		Route::post('task-delete',[TaskApiController::class,'taskDelete']);

    });
});
