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

namespace YooKassa\Model\Payment\PaymentMethod\B2b\Sberbank;

use InvalidArgumentException;

/**
 * Класс, представляющий модель PaymentMethodDataCash.
 *
 * Фабрика создания объекта данных об НДС.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class VatDataFactory
{
    private array $typeClassMap = [
        VatDataType::UNTAXED => 'UntaxedVatData',
        VatDataType::CALCULATED => 'CalculatedVatData',
        VatDataType::MIXED => 'MixedVatData',
    ];

    /**
     * Фабричный метод создания данных об НДС по типу.
     *
     * @param string|null $type Тип данных об НДС
     *
     * @return AbstractVatData
     */
    public function factory(?string $type = null, ?array $data = []): AbstractVatData
    {
        if (!is_string($type)) {
            throw new InvalidArgumentException('Invalid payment type value in payment factory');
        }
        if (!array_key_exists($type, $this->typeClassMap)) {
            throw new InvalidArgumentException('Invalid payment data type "' . $type . '"');
        }
        $className = __NAMESPACE__ . '\\' . $this->typeClassMap[$type];

        return new $className($data);
    }

    /**
     * Фабричный метод создания данных об НДС из массива.
     *
     * @param array|null $data Массив данных об НДС
     * @param string|null $type Тип данных об НДС
     */
    public function factoryFromArray(?array $data = [], ?string $type = null): AbstractVatData
    {
        if (null === $type) {
            if (array_key_exists('type', $data)) {
                $type = $data['type'];
                unset($data['type']);
            } else {
                throw new InvalidArgumentException(
                    'Parameter type not specified in PaymentDataFactory.factoryFromArray()'
                );
            }
        }

        return $this->factory($type, $data);
    }
}
