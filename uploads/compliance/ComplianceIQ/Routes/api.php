<?php

use Illuminate\Support\Facades\Route;

Route::post('/explain', 'ExplainController@explain')->name('explain');
