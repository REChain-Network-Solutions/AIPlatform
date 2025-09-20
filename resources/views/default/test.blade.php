@extends('panel.layout.app', ['disable_tblr' => true, 'disable_titlebar' => true])
@section('title', __('Overview'))

@section('content')
    <script
        defer
        src="https://magicai.test/vendor/chatbot/js/external-chatbot.js"
        data-chatbot-uuid="f9f09630-a8c8-49b2-bf12-461bd6775912"
        data-iframe-width="420"
        data-iframe-height="745"
        data-language="en"
    ></script>
@endsection
