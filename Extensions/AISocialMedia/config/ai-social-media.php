<?php

return [
    'instagram' => [
        'app_id'       => '',
        'app_secret'   => '',
        'base_url'     => 'https://www.facebook.com',
        'api_url'      => 'https://graph.facebook.com',
        'redirect_uri' => '/oauth/callback/instagram',
        'api_version'  => 'v18.0',
        'scopes'       => [
            'pages_manage_posts',
            'pages_show_list',
            'pages_read_user_content',
            'pages_read_engagement',
            'read_insights',

            'ads_management',
            'business_management',
            'instagram_basic',
            'instagram_content_publish',
        ],
        'requirements' => [
            'text' => [
                'limit' => 1000,
            ],
            'images' => [
                'limit'  => 10,
                'width'  => 640,
                'height' => 640,
                'size'   => 1024, // in kb
            ],
            'videos' => [
                'limit'  => 10,
                'width'  => 640,
                'height' => 640,
                'size'   => 1024, // in kb
            ],
        ],
    ],
];
