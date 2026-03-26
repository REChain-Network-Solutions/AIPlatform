<?php

use Illuminate\Support\Facades\Route;

Route::get('/reports', 'ComplianceReportController@index')->name('reports.index');
Route::get('/reports/create', 'ComplianceReportController@create')->name('reports.create');
Route::post('/reports', 'ComplianceReportController@store')->name('reports.store');
Route::get('/reports/{report}', 'ComplianceReportController@show')->name('reports.show');
Route::get('/reports/{report}/edit', 'ComplianceReportController@edit')->name('reports.edit');
Route::put('/reports/{report}', 'ComplianceReportController@update')->name('reports.update');
Route::delete('/reports/{report}', 'ComplianceReportController@destroy')->name('reports.destroy');

Route::post('/reports/{report}/export', 'ComplianceReportController@export')->name('reports.export');
Route::post('/reports/{report}/signoff', 'ComplianceReportController@signoff')->name('reports.signoff');
Route::post('/reports/{report}/annotate', 'ComplianceReportController@annotate')->name('reports.annotate');

Route::get('/logs', 'ComplianceLogController@index')->name('logs.index');
