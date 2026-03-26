<?php

namespace Modules\Feedback\Http\Requests;

use App\Http\Requests\CoreRequest;

class UpdateFeedbackType extends CoreRequest
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
            'type' => 'required|unique:feedback_types,type,'.$this->route('feedbackType').',id,company_id,' . company()->id,
        ];
    }

}
