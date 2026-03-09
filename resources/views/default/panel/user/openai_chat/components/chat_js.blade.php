<script>
    var chatid = @json($list)[0]?.id;
    $(`#chat_${chatid}`).addClass('active').siblings().removeClass('active');
    const guest_id = document.getElementById("guest_id")?.value;
    const guest_search = document.getElementById("guest_search")?.value;
    const guest_search_id = document.getElementById("guest_search_id")?.value;
    const guest_event_id = document.getElementById("guest_event_id")?.value;
    const guest_look_id = document.getElementById("guest_look_id")?.value;
    const guest_product_id = document.getElementById("guest_product_id")?.value;
    const stream_type = '{!! $settings_two->openai_default_stream_server !!}';
    const category = @json($category);
    const openai_model = '{!! $setting->openai_default_model !!}';
    const prompt_prefix = document.getElementById("prompt_prefix")?.value;

    let messages = [];
    let training = [];

    @if ($chat_completions != null)
        training = @json($chat_completions);
    @endif

    messages.push({
        role: "assistant",
        content: prompt_prefix
    });

    @if ($lastThreeMessage != null)
        @foreach ($lastThreeMessage as $entry)
            message = {
                role: "user",
                content: @json($entry->input)
            };
            messages.push(message);
            message = {
                role: "assistant",
                content: @json(str_replace('/http', 'http', $entry->output))
            };
            messages.push(message);
        @endforeach
    @endif
</script>

<script src="{{ custom_theme_url('/assets/js/panel/openai_chat.js?v=' . time()) }}"></script>

@if (count($list) == 0 && $category->slug != 'ai_pdf')
    @php
        $template = null;
        $currentUrl = url()->current();
        $currentPath = trim(parse_url($currentUrl, PHP_URL_PATH) ?: '');
        if (
            \App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-chat-pro') &&
            (str_starts_with($currentPath, '/chat') || str_starts_with($currentPath, '/dashboard/user/openai/chat/pro/') || !auth()->check())
        ) {
            $template = 'chatpro';
        }
        if (
            \App\Helpers\Classes\MarketplaceHelper::isRegistered('social-media-agent') &&
            (str_starts_with($currentPath, '/dashboard/user/social-media/agent/chat') || !auth()->check())
        ) {
            $template = 'social-media-agent';
        }
    @endphp
    <script>
        window.addEventListener("load", (event) => {
            return startNewChat(
                {{ $category->id }},
                '{{ LaravelLocalization::getCurrentLocale() }}',
                '{{ $template }}'
            );
        });
    </script>
@endif

<script>
    function getProductByBrand(brand_id) {
        var brand_id = brand_id;
        var product_element = $('#brand_voice_prod');
        if (brand_id == '') {
            product_element.empty();
            product_element.append('<option value="">Select Product</option>');
        } else {
            $.ajax({
                url: '/dashboard/user/brand/get-products/' + brand_id,
                type: 'get',
                success: function(response) {
                    product_element.empty();
                    if (response.length == 0) {
                        product_element.append('<option value="">Select Product</option>');
                    } else {
                        $.each(response, function(index, value) {
                            product_element.append('<option value="' + value.id + '">' + value.name + '</option>');
                        });
                    }
                }
            });
        }
    }

    function setBrandVoice() {
        toastr.success('Brand voice selected');
        $('#brandVoiceModal').modal('hide');
    }
</script>

@includeFirst(['chat-share::share-script-include', 'panel.user.openai_chat.includes.share-script-include', 'vendor.empty'])
