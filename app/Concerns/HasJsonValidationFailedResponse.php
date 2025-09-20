<?php

namespace App\Concerns;

use Illuminate\Contracts\Validation\Validator;
use Illuminate\Http\Exceptions\HttpResponseException;
use Illuminate\Http\JsonResponse;

trait HasJsonValidationFailedResponse
{
    // Handle a failed validation attempt and return a JSON response.
    protected function failedValidation(Validator $validator): JsonResponse
    {
        $firstError = $validator->errors()->first();

        throw new HttpResponseException(
            response()->json([
                'status'  => 'error',
                'message' => $firstError,
            ], JsonResponse::HTTP_UNPROCESSABLE_ENTITY)
        );
    }
}
