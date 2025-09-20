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

namespace YooKassa\Model\Payment;

use Exception;
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
 *
 * @property string $accountId Идентификатор магазина
 * @property string $account_id Идентификатор магазина
 * @property string $gatewayId Идентификатор шлюза
 * @property string $gateway_id Идентификатор шлюза
 */
class Recipient extends AbstractObject implements RecipientInterface
{
    /**
     * @var string|null Идентификатор магазина
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_account_id = null;

    /**
     * Идентификатор субаккаунта. Используется для разделения потоков платежей в рамках одного аккаунта.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_gateway_id = null;

    /**
     * Возвращает идентификатор магазина.
     *
     * @return string|null Идентификатор магазина
     */
    public function getAccountId(): ?string
    {
        return $this->_account_id;
    }

    /**
     * Устанавливает идентификатор магазина.
     *
     * @param string|null $account_id Идентификатор магазина
     *
     * @throws Exception Выбрасывается если было передано не строковое значение
     * @return self
     */
    public function setAccountId(?string $account_id): self
    {
        $this->_account_id = $this->validatePropertyValue('_account_id', $account_id);
        return $this;
    }

    /**
     * Возвращает идентификатор субаккаунта.
     *
     * @return string|null Идентификатор субаккаунта. Используется для разделения потоков платежей в рамках одного аккаунта.
     */
    public function getGatewayId(): ?string
    {
        return $this->_gateway_id;
    }

    /**
     * Устанавливает идентификатор субаккаунта.
     *
     * @param string|null $gateway_id Идентификатор субаккаунта. Используется для разделения потоков платежей в рамках одного аккаунта.
     *
     * @throws Exception
     * @return self
     */
    public function setGatewayId(?string $gateway_id = null): self
    {
        $this->_gateway_id = $this->validatePropertyValue('_gateway_id', $gateway_id);
        return $this;
    }
}
