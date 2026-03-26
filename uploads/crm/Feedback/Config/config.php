<?php

return [
    'name' => 'Feedback'    ,
    'menu_table' => env('FEEDBACK_MENU_TABLE','menus'),
    'ai' => require __DIR__.'/ai.php',
];
