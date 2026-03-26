<?php

use Illuminate\Support\Facades\Route;


//Admin routes
Route::group(
    ['prefix' => 'admin', 'as' => 'admin.audit-log.', 'namespace' => 'Admin', 'middleware' => ['auth', 'role:admin']],
    function () {
        Route::get('audit-log/leave', 'LeaveLogsController@index')->name('leave');
        Route::get('audit-log/leave-export', 'LeaveLogsController@export')->name('leave-export');
        Route::get('audit-log/user', 'AuditLogController@user')->name('user');
        Route::get('audit-log/task', 'AuditLogController@task')->name('task');
        Route::get('audit-log/project', 'AuditLogController@project')->name('project');
        Route::get('log-activities', 'AuditLogController@index')->name('log-activities');
        Route::get('audit-log/attendance', 'AuditLogController@attendance')->name('attendance');
        Route::get('audit-log/incident', 'AuditLogController@incident')->name('incident');
        Route::get('log-activities-export', 'AuditLogController@export')->name('log-activities.export');
        Route::get('audit-log/attendance-export', 'AuditLogController@AttendanceExport')->name('attendance-export');
    }
);

//Member routes
Route::group(
    ['prefix' => 'member', 'as' => 'member.audit-log.', 'namespace' => 'Member'],
    function () {
        Route::get('audit-log/leave', 'LeaveLogsController@index')->name('leave');
        Route::get('audit-log/leave-export', 'LeaveLogsController@export')->name('leave-export');
        Route::get('audit-log/user', 'AuditLogController@user')->name('user');
        Route::get('audit-log/task', 'AuditLogController@task')->name('task');
        Route::get('audit-log/project', 'AuditLogController@project')->name('project');
        Route::get('log-activities', 'AuditLogController@index')->name('log-activities');
        Route::get('audit-log/attendance', 'AuditLogController@attendance')->name('attendance');
        Route::get('audit-log/incident', 'AuditLogController@incident')->name('incident');
        Route::get('log-activities-export', 'AuditLogController@export')->name('log-activities.export');
        Route::get('audit-log/attendance-export', 'AuditLogController@AttendanceExport')->name('attendance-export');
    }
);
