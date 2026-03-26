<?php

namespace Modules\Feedback\Http\Requests;

use App\Http\Requests\CoreRequest;

class StoreFeedbackChannel extends CoreRequest
{

    /**
     * Determine if the user is authorized to make this request.
     *
     * @return bool
     */
    public function authorize()
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array
     */
    public function rules()
    {
        return [
            'channel_name' => 'required|unique:feedback_channels,channel_name,null,id,company_id,' . company()->id
        ];
    }

}
