<?php

Route::group(['prefix' => config('elseyyid-location.prefix'), 'middleware' => config('elseyyid-location.middlewares') ,'as' => 'elseyyid.translations.'], function(){

    Route::get('home', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@index')->name('home');
    Route::get('lang/{lang}', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@lang')->name('lang');
    Route::get('lang/generateJson/{lang}', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@generateJson')->name('lang.generateJson');
    Route::get('newLang', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@newLang')->name('lang.newLang');
    Route::get('newString', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@newString')->name('lang.newString');
    Route::get('search', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@search')->name('lang.search');
    Route::get('string/{code}', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@string')->name('lang.string');
    Route::get('publish-all', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@publishAll')->name('lang.publishAll');
});
Route::post('translations/lang/update/{id}', '\Elseyyid\LaravelJsonLocationsManager\Controllers\HomeController@update')->name('elseyyid.translations.lang.update');
