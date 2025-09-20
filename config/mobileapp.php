<?php

return [
    'app_name'       => env('MOBILE_APP_NAME', env('APP_NAME', 'Laravel Mobile App')),
    'developer_name' => env('MOBILE_APP_DEVELOPER_NAME', null),
    'package_id'     => env('MOBILE_APP_PACKAGE_ID', env('APPLE_BUNDLE_ID', 'com.liquidthemes.magicai')),
    'mailto'         => env('MOBILE_APP_MAILTO', null),
];
