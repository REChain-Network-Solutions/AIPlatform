<?php

ApiRoute::group(['namespace' => 'Modules\AuditLog\Http\Controllers', 'middleware' => 'api.auth'], function() {
    // ApiRoute::resource('/auditlog', 'AuditLogController');
});