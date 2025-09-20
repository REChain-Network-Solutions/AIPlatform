<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Common\Errors;

use InvalidArgumentException;

/**
 * Класс, представляющий модель ErrorFactory.
 *
 * Фабрика создания объекта ошибки из массива.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class ErrorFactory
{
    private array $typeClassMap = [
        ErrorCode::ERROR => 'ErrorCommon',
        ErrorCode::INVALID_REQUEST => 'ErrorBadRequest',
        ErrorCode::INVALID_CREDENTIALS => 'ErrorInvalidCredentials',
        ErrorCode::FORBIDDEN => 'ErrorForbidden',
        ErrorCode::NOT_FOUND => 'ErrorNotFound',
        ErrorCode::GONE => 'ErrorGone',
        ErrorCode::TOO_MANY_REQUESTS => 'ErrorTooManyRequests',
        ErrorCode::INTERNAL_SERVER_ERROR => 'ErrorInternalServerError',
        ErrorCode::UNKNOWN => 'ErrorUnknown',
    ];

    /**
     * Фабричный метод создания объекта ошибки по типу.
     *
     * @param string|null $code Код ошибки
     *
     * @return AbstractError
     */
    public function factory(?string $code): AbstractError
    {
        if (!array_key_exists($code, $this->typeClassMap)) {
            $code = ErrorCode::UNKNOWN;
        }
        $className = __NAMESPACE__ . '\\' . $this->typeClassMap[$code];

        return new $className();
    }

    /**
     * Фабричный метод создания объекта ошибки из массива.
     *
     * @param array $data Массив данных ошибки
     * @param null|string $code Код ошибки
     */
    public function factoryFromArray(array $data, ?string $code = null): AbstractError
    {
        if (null === $code) {
            if (array_key_exists('code', $data)) {
                $code = $data['code'];
                unset($data['code']);
            } else {
                $code = ErrorCode::UNKNOWN;
            }
        }

        $errorData = $this->factory($code);
        $errorData->fromArray($data);

        return $errorData;
    }
}
