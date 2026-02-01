<?php

namespace App\Extensions\ChatSetting\System\Http\Controllers\Chatbot;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Http\Requests\Chatbot\ChatbotRequest;
use App\Models\Chatbot\Chatbot;
use App\Models\Chatbot\ChatbotData;
use App\Models\Chatbot\ChatbotDataVector;
use Illuminate\Contracts\View\View;
use Illuminate\Http\RedirectResponse;

class ChatbotController extends Controller
{
    public function index(): View
    {
        abort_if(setting('chat_setting_for_customer', '1') == '0', 404);

        return view('chat-setting::chatbot.index', [
            'title' => trans('Chatbot Training'),
            'items' => Chatbot::query()
                ->where('user_id', auth()->id())
                ->paginate(10),
        ]);
    }

    public function create(): RedirectResponse
    {
        abort_if(setting('chat_setting_for_customer', '1') == '0', 404);

        $item = Chatbot::query()->create([
            'user_id' => auth()->id(),
            'title'   => 'Untitled chatbot',
            'model'   => Helper::setting('openai_default_model'),
        ]);

        return to_route('dashboard.user.chat-setting.chatbot.show', $item->getAttribute('id'));
    }

    public function store(ChatbotRequest $request)
    {
        abort_if(setting('chat_setting_for_customer', '1') == '0', 404);

        if (Helper::appIsDemo()) {
            return back()->with([
                'type'    => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $data = $request->validated();

        if ($request->hasFile('logo')) {
            $data['image'] = $request->file('logo')->store('chatbot');
        }

        $data['user_id'] = auth()->id();

        Chatbot::query()->create($data);

        return to_route('dashboard.user.chat-setting.chatbot.index')->with('success', trans('Chatbot Template Created Successfully'));
    }

    public function show(Chatbot $chatbot)
    {
        if ($chatbot->user_id != auth()->id()) {
            abort(403, 'Unauthorized');
        }

        return view('chat-setting::chatbot.training', [
            'title'  => trans('Chatbot Training'),
            'item'   => $chatbot,
            'data'   => $chatbot->data()->get(),
            'action' => route('dashboard.user.chat-setting.chatbot.update', $chatbot),
        ]);
    }

    public function edit(Chatbot $chatbot)
    {
        if ($chatbot->user_id != auth()->id()) {
            abort(403, 'Unauthorized');
        }

        return view('chat-setting::chatbot.form', [
            'title'  => trans('Edit Chatbot'),
            'method' => 'put',
            'action' => route('dashboard.user.chat-setting.chatbot.update', $chatbot),
            'item'   => $chatbot,
        ]);
    }

    public function update(ChatbotRequest $request, Chatbot $chatbot)
    {
        if (Helper::appIsDemo()) {
            if ($request->ajax()) {
                return response()->json([
                    'status'  => 'error',
                    'message' => trans('This feature is disabled in demo mode.'),
                ]);
            }

            return back()->with([
                'type'    => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $data = $request->validated();

        if ($request->hasFile('logo')) {
            $data['image'] = $request->file('logo')->store('chatbot');
        }

        $chatbot->update($data);

        if ($request->ajax()) {
            return response()->json([
                'message' => trans('Chatbot Updated Successfully'),
                'status'  => 'success',
            ]);
        }

        return back()->with([
            'type'    => 'success',
            'message' => trans('Chatbot Updated Successfully'),
        ]);

    }

    public function destroy(Chatbot $chatbot): \Illuminate\Http\JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $chatbotId = $chatbot->getAttribute('id');

        ChatbotData::query()->where('chatbot_id', $chatbotId)->delete();

        ChatbotDataVector::query()->where('chatbot_id', $chatbotId)->delete();

        $chatbot->delete();

        return response()->json([
            'message'    => trans('Chatbot Deleted Successfully'),
            'reload'     => true,
            'setTimeOut' => 1000,
        ]);
    }
}
