<?php

namespace App\Extensions\ChatSetting\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\Chatbot\Chatbot;
use App\Models\ChatCategory;
use App\Models\OpenaiGeneratorChatCategory;
use App\Services\Assistant\AssistantService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Str;

class ChatTemplateController extends Controller
{
    public function index()
    {
        abort_if(setting('chat_setting_for_customer', '1') == '0', 404);

        $list = OpenaiGeneratorChatCategory::query()
            ->where('user_id', auth()->id())
            ->where('slug', '<>', 'ai_vision')
            ->where('slug', '<>', 'ai_pdf')
            ->orderBy('name', 'asc')
            ->get();

        return view('chat-setting::template.index', compact('list'));
    }

    public function create()
    {
        abort_if(setting('chat_setting_for_customer', '1') == '0', 404);

        $categoryList = ChatCategory::query()
            ->where('user_id', auth()->id())
            ->get();

        $chatbots = Chatbot::query()->where('user_id', auth()->id())->get();

        $assistantService = new AssistantService;

        $assistants = $assistantService->listAssistant()['data'] ?? [];

        return view('chat-setting::template.form', [
            'method'       => 'POST',
            'title'        => 'Create Chat Template',
            'action'       => route('dashboard.user.chat-setting.chat-template.store'),
            'categoryList' => $categoryList,
            'chatbots'     => $chatbots,
            'assistants'   => $assistants,
            'template'     => new OpenaiGeneratorChatCategory,
        ]);

    }

    public function store(Request $request)
    {
        $request->validate([
            'name' => 'required',
        ]);

        $template = new OpenaiGeneratorChatCategory;

        if ($request->hasFile('avatar')) {
            $path = 'upload/images/chatbot/';
            $image = $request->file('avatar');
            $image_name = Str::random(4) . '-' . Str::slug($request->name) . '-avatar.' . $image->getClientOriginalExtension();

            // Resim uzantı kontrolü
            $imageTypes = ['jpg', 'jpeg', 'png', 'svg', 'webp'];
            if (! in_array(Str::lower($image->getClientOriginalExtension()), $imageTypes)) {
                $data = [
                    'errors' => ['The file extension must be jpg, jpeg, png, webp or svg.'],
                ];

                return response()->json($data, 419);
            }

            $image->move($path, $image_name);

            $template->image = $path . $image_name;
        }

        $template->name = $request->name;
        $template->category = $request->chat_category ?? '';
        $template->user_id = auth()->id();
        $template->slug = Str::slug($request->name) . '-' . Str::random(5);
        $template->short_name = $request->short_name;
        $template->description = $request->description;
        if (Schema::hasColumn('openai_chat_category', 'instructions')) {
            $template->instructions = $request->instructions;
        }

        if (Schema::hasColumn('openai_chat_category', 'first_message')) {
            $template->first_message = $request->first_message;
        }

        $template->role = $request->role;
        $template->human_name = $request->human_name;
        $template->helps_with = $request->helps_with;
        $template->color = $request->color;
        $template->plan = 'regular';
        $template->chat_completions = $request->chat_completions;
        $template->prompt_prefix = 'As a ' . $request->role . ', ';
        $template->chatbot_id = $request->chatbot_id;
        $template->assistant = $request->assistant;
        $template->save();

        return redirect()->route('dashboard.user.chat-setting.chat-template.index')->with([
            'type'    => 'success',
            'message' => 'Chat Template created successfully.',
        ]);
    }

    public function edit(OpenaiGeneratorChatCategory $chatTemplate)
    {
        if ($chatTemplate->user_id != auth()->id()) {
            abort(403, 'Unauthorized');
        }

        $categoryList = ChatCategory::query()->where('user_id', auth()->id())->get();

        $chatbots = Chatbot::query()->where('user_id', auth()->id())->get();

        $assistantService = new AssistantService;
        $assistants = $assistantService->listAssistant()['data'] ?? [];

        return view('chat-setting::template.form', [
            'title'        => 'Create Chat Template',
            'action'       => route('dashboard.user.chat-setting.chat-template.update', $chatTemplate),
            'categoryList' => $categoryList,
            'chatbots'     => $chatbots,
            'method'       => 'PUT',
            'template'     => $chatTemplate,
            'assistants'   => $assistants,
        ]);
    }

    public function update(Request $request, OpenaiGeneratorChatCategory $chatTemplate)
    {
        $request->validate([
            'name' => 'required',
        ]);

        $template = $chatTemplate;

        if ($request->hasFile('avatar')) {
            $path = 'upload/images/chatbot/';
            $image = $request->file('avatar');
            $image_name = Str::random(4) . '-' . Str::slug($request->name) . '-avatar.' . $image->getClientOriginalExtension();

            // Resim uzantı kontrolü
            $imageTypes = ['jpg', 'jpeg', 'png', 'svg', 'webp'];
            if (! in_array(Str::lower($image->getClientOriginalExtension()), $imageTypes)) {
                $data = [
                    'errors' => ['The file extension must be jpg, jpeg, png, webp or svg.'],
                ];

                return response()->json($data, 419);
            }

            $image->move($path, $image_name);

            $template->image = $path . $image_name;
        }

        $template->name = $request->name;
        $template->category = $request->chat_category ?? '';
        $template->slug = Str::slug($request->name) . '-' . Str::random(5);
        $template->short_name = $request->short_name;
        $template->description = $request->description;
        if (Schema::hasColumn('openai_chat_category', 'instructions')) {
            $template->instructions = $request->instructions;
        }

        if (Schema::hasColumn('openai_chat_category', 'first_message')) {
            $template->first_message = $request->first_message;
        }

        $template->role = $request->role;
        $template->human_name = $request->human_name;
        $template->helps_with = $request->helps_with;
        $template->color = $request->color;
        $template->plan = 'regular';
        $template->chat_completions = $request->chat_completions;
        $template->prompt_prefix = 'As a ' . $request->role . ', ';
        $template->chatbot_id = $request->chatbot_id;
        $template->assistant = $request->assistant;
        $template->save();

        return redirect()->route('dashboard.user.chat-setting.chat-template.index')->with([
            'type'    => 'success',
            'message' => 'Chat Template created successfully.',
        ]);
    }

    public function destroy(OpenaiGeneratorChatCategory $chatTemplate): \Illuminate\Http\JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $chatTemplate->delete();

        return response()->json([
            'message'    => trans('Template Deleted Successfully'),
            'reload'     => true,
            'setTimeOut' => 1000,
        ]);
    }
}
