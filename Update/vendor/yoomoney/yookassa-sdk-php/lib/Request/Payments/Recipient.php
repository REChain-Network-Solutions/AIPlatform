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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Recipient.
 *
 * Получатель платежа. Нужен, если вы разделяете потоки платежей в рамках одного аккаунта или создаете платеж в адрес другого аккаунта.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $gateway_id Идентификатор субаккаунта. Используется для разделения потоков платежей в рамках одного аккаунта.
*/
class Recipient extends AbstractObject
{
    /**
     * Идентификатор субаккаунта. Используется для разделения потоков платежей в рамках одного аккаунта.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_gateway_id = null;

    /**
     * Возвращает gateway_id.
     *
     * @return string|null
     */
    public function getGatewayId(): ?string
    {
        return $this->_gateway_id;
    }

    /**
     * Устанавливает gateway_id.
     *
     * @param string|null $gateway_id Идентификатор субаккаунта. Используется для разделения потоков платежей в рамках одного аккаунта.
     *
     * @return self
     */
    public function setGatewayId(?string $gateway_id = null): self
    {
        $this->_gateway_id = $this->validatePropertyValue('_gateway_id', $gateway_id);
        return $this;
    }

}

