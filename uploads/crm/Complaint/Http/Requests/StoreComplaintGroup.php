<?php

namespace Modules\Complaint\Http\Requests;

use App\Http\Requests\CoreRequest;

class StoreComplaintGroup extends CoreRequest
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
            'group_name' => 'required|unique:complaint_groups,group_name,null,id,company_id,' . company()->id
        ];
    }

}
