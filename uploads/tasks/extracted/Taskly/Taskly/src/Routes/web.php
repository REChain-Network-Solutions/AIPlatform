<?php

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

use Illuminate\Support\Facades\Route;
use Workdo\Taskly\Http\Controllers\BugStageController;
use Workdo\Taskly\Http\Controllers\DashboardController;
use Workdo\Taskly\Http\Controllers\ProjectController;
use Workdo\Taskly\Http\Controllers\ProjectReportController;
use Workdo\Taskly\Http\Controllers\StageController;

Route::middleware(['web','auth','verified','PlanModuleCheck:Taskly'])->group(function ()
{
    Route::get('dashboard/taskly',[DashboardController::class,'index'])->name('taskly.dashboard');

    Route::get('/project/copy/{id}',[ProjectController::class,'copyproject'])->name('project.copy');
    Route::post('/project/copy/store/{id}',[ProjectController::class,'copyprojectstore'])->name('project.copy.store');

    Route::resource('projects', ProjectController::class);
    Route::resource('stages', StageController::class);

    Route::get('projects-list', [ProjectController::class,'List'])->name('projects.list');

    //project import
    Route::get('project/import/export', [ProjectController::class,'fileImportExport'])->name('project.file.import');
    Route::post('project/import', [ProjectController::class,'fileImport'])->name('project.import');
    Route::get('project/import/modal', [ProjectController::class,'fileImportModal'])->name('project.import.modal');
    Route::post('project/data/import/', [ProjectController::class,'projectImportdata'])->name('project.import.data');

    //project Setting
    Route::get('project/setting/{id}', [ProjectController::class,'CopylinkSetting'])->name('project.setting');
    Route::post('project/setting/save{id}', [ProjectController::class,'CopylinkSettingSave'])->name('project.setting.save');

    Route::post('send-mail', [ProjectController::class,'sendMail'])->name('send.mail');
    // Task Board
    Route::get('projects/{id}/task-board',[ProjectController::class,'taskBoard'])->name('projects.task.board');
    Route::get('projects/{id}/calendar',[ProjectController::class,'calendar'])->name('projects.calendar');

    Route::get('projects/{id}/task-board/create',[ProjectController::class,'taskCreate'])->name('tasks.create');
    Route::post('projects/{id}/task-board/store',[ProjectController::class,'taskStore'])->name('tasks.store');
    Route::post('projects/{id}/task-board/order-update',[ProjectController::class,'taskOrderUpdate'])->name('tasks.update.order');
    Route::get('projects/{id}/task-board/edit/{tid}',[ProjectController::class,'taskEdit'])->name('tasks.edit');
    Route::post('projects/{id}/task-board/{tid}/update',[ProjectController::class,'taskUpdate'])->name('tasks.update');
    Route::delete('projects/{id}/task-board/{tid}',[ProjectController::class,'taskDestroy'])->name('tasks.destroy');
    Route::get('projects/{id}/task-board/{tid}/{cid?}',[ProjectController::class,'taskShow'])->name('tasks.show');
    Route::get('projects/{id}/task-board-list', [ProjectController::class,'TaskList'])->name('projecttask.list');
    Route::post('projects/task-member/{id}/{cid?}', [ProjectController::class,'TaskMember'])->name('tasks.members');

    // Gantt Chart
    Route::get('projects/{id}/gantt/{duration?}',[ProjectController::class,'gantt'])->name('projects.gantt');
    Route::post('projects/{id}/gantt',[ProjectController::class,'ganttPost'])->name('projects.gantt.post');

    // finance page
    Route::get('projects/{id}/proposal',[ProjectController::class,'proposal'])->name('projects.proposal');
    Route::get('projects/{id}/invoice',[ProjectController::class,'invoice'])->name('projects.invoice');



    // bug report
    Route::get('projects/{id}/bug_report',[ProjectController::class,'bugReport'])->name('projects.bug.report');
    Route::get('projects/{id}/bug_report/create',[ProjectController::class,'bugReportCreate'])->name('projects.bug.report.create');
    Route::post('projects/{id}/bug_report',[ProjectController::class,'bugReportStore'])->name('projects.bug.report.store');
    Route::post('projects/{id}/bug_report/order-update',[ProjectController::class,'bugReportOrderUpdate'])->name('projects.bug.report.update.order');
    Route::get('projects/{id}/bug_report/{bid}/show',[ProjectController::class,'bugReportShow'])->name('projects.bug.report.show');
    Route::get('projects/{id}/bug_report/{bid}/edit',[ProjectController::class,'bugReportEdit'])->name('projects.bug.report.edit');
    Route::post('projects/{id}/bug_report/{bid}/update',[ProjectController::class,'bugReportUpdate'])->name('projects.bug.report.update');
    Route::delete('projects/{id}/bug_report/{bid}',[ProjectController::class,'bugReportDestroy'])->name('projects.bug.report.destroy');
    Route::get('projects/{id}/bug_report-list', [ProjectController::class,'BugList'])->name('projectbug.list');


    Route::get('projects/invite/{id}',[ProjectController::class,'popup'])->name('projects.invite.popup');
    Route::get('projects/share/{id}',[ProjectController::class,'sharePopup'])->name('projects.share.popup');
    Route::get('projects/share/vender/{id}',[ProjectController::class,'sharePopupVender'])->name('projects.share.vender.popup');
    Route::post('projects/share/vender/store/{id}',[ProjectController::class,'sharePopupVenderStore'])->name('projects.share.vender');
    Route::get('projects/milestone/{id}',[ProjectController::class,'milestone'])->name('projects.milestone');
    Route::post('projects/{id}/file',[ProjectController::class,'fileUpload'])->name('projects.file.upload');
    Route::post('projects/share/{id}',[ProjectController::class,'share'])->name('projects.share');


    // stages.index
    // project
    Route::get('projects/milestone/{id}',[ProjectController::class,'milestone'])->name('projects.milestone');
    Route::post('projects/milestone/{id}/store',[ProjectController::class,'milestoneStore'])->name('projects.milestone.store');
    Route::get('projects/milestone/{id}/show',[ProjectController::class,'milestoneShow'])->name('projects.milestone.show');
    Route::get('projects/milestone/{id}/edit',[ProjectController::class,'milestoneEdit'])->name('projects.milestone.edit');
    Route::post('projects/milestone/{id}/update',[ProjectController::class,'milestoneUpdate'])->name('projects.milestone.update');
    Route::delete('projects/milestone/{id}',[ProjectController::class,'milestoneDestroy'])->name('projects.milestone.destroy');
    Route::delete('projects/{id}/file/delete/{fid}',[ProjectController::class,'fileDelete'])->name('projects.file.delete');


    Route::post('projects/invite/{id}/update',[ProjectController::class,'invite'])->name('projects.invite.update');

    Route::resource('bugstages', BugStageController::class);


    Route::post('projects/{id}/comment/{tid}/file/{cid?}',[ProjectController::class,'commentStoreFile'])->name('comment.store.file');
    Route::delete('projects/{id}/comment/{tid}/file/{fid}',[ProjectController::class,'commentDestroyFile'])->name('comment.destroy.file');
    Route::post('projects/{id}/comment/{tid}/{cid?}',[ProjectController::class,'commentStore'])->name('comment.store');
    Route::delete('projects/{id}/comment/{tid}/{cid}',[ProjectController::class,'commentDestroy'])->name('comment.destroy');
    Route::post('projects/{id}/sub-task/update/{stid}',[ProjectController::class,'subTaskUpdate'])->name('subtask.update');
    Route::post('projects/{id}/sub-task/{tid}/{cid?}',[ProjectController::class,'subTaskStore'])->name('subtask.store');
    Route::delete('projects/{id}/sub-task/{stid}',[ProjectController::class,'subTaskDestroy'])->name('subtask.destroy');

    Route::post('projects/{id}/bug_comment/{tid}/file/{cid?}',[ProjectController::class,'bugStoreFile'])->name('bug.comment.store.file');
    Route::delete('projects/{id}/bug_comment/{tid}/file/{fid}',[ProjectController::class,'bugDestroyFile'])->name('bug.comment.destroy.file');
    Route::post('projects/{id}/bug_comment/{tid}/{cid?}',[ProjectController::class,'bugCommentStore'])->name('bug.comment.store');
    Route::delete('projects/{id}/bug_comment/{tid}/{cid}',[ProjectController::class,'bugCommentDestroy'])->name('bug.comment.destroy');
    Route::delete('projects/{id}/client/{uid}',[ProjectController::class,'clientDelete'])->name('projects.client.delete');
    Route::delete('projects/{id}/user/{uid}',[ProjectController::class,'userDelete'])->name('projects.user.delete');
    Route::delete('projects/{id}/vendor/{uid}',[ProjectController::class,'vendorDelete'])->name('projects.vendor.delete');

    // Project Report
    Route::resource('project_report', ProjectReportController::class);

    Route::post('reports-quarterly-cashflow/{id}', [ProjectReportController::class, 'quarterlyCashflow'])->name('projectreport.quarterly.cashflow');

    Route::post('project_report_data',[ProjectReportController::class,'ajax_data'])->name('projects.ajax');
    Route::post('project_report/tasks/{id}',[ProjectReportController::class,'ajax_tasks_report'])->name('tasks.report.ajaxdata');
});
Route::middleware(['web'])->group(function ()
{
    Route::get('projects/{id}/file/{fid}',[ProjectController::class,'fileDownload'])->name('projects.file.download');

    Route::post('project/password/check/{id}/{lang?}', [ProjectController::class,'PasswordCheck'])->name('project.password.check');
    Route::get('project/shared/link/{id}/{lang?}', [ProjectController::class,'ProjectSharedLink'])->name('project.shared.link');
    Route::get('projects/{id}/link/task/show/{tid}/',[ProjectController::class,'ProjectLinkTaskShow'])->name('Project.link.task.show');
    Route::get('projects/{id}/link/bug_report/{bid}/show',[ProjectController::class,'ProjectLinkbugReportShow'])->name('projects.link.bug.report.show');
});
