<?php

namespace Modules\Complaint\Http\Requests;

use App\Http\Requests\CoreRequest;

class UpdateComplaintType extends CoreRequest
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
            'type' => 'required|unique:complaint_types,type,'.$this->route('complaintType').',id,company_id,' . company()->id,
        ];
    }

}
