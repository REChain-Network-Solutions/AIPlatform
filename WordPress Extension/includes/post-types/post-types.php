<?php

add_action( 'init', function( ){
    include __DIR__ . '/attachments.php';
    include __DIR__ . '/chat.php';
    include __DIR__ . '/chat-pdf.php';
    include __DIR__ . '/vision.php';
    include __DIR__ . '/chatbot.php';
    include __DIR__ . '/documents.php';
} );