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

namespace YooKassa\Request\Payments\PaymentData;

use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentMethodDataAlfabank.
 *
 * Данные для оплаты через Альфа-Клик (или Альфа-Молнию).
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @deprecated Будет удален в следующих версиях
 *
 * @property string $login Логин пользователя в Альфа-Клике. Обязателен для сценария [External](/developers/payment-acceptance/getting-started/payment-process#external).
 */
class PaymentDataAlfabank extends AbstractPaymentData
{
    /**
     * @var string|null Имя пользователя в Альфа-Клике
     */
    #[Assert\Type('string')]
    private ?string $_login = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(PaymentMethodType::ALFABANK);
    }

    /**
     * Возвращает имя пользователя в Альфа-Клике.
     *
     * @return string|null Имя пользователя в Альфа-Клике
     */
    public function getLogin(): ?string
    {
        return $this->_login;
    }

    /**
     * Устанавливает имя пользователя в Альфа-Клике.
     *
     * @param string|null $login Имя пользователя в Альфа-Клике
     *
     * @return self
     */
    public function setLogin(?string $login): self
    {
        $this->_login = $this->validatePropertyValue('_login', $login);
        return $this;
    }
}
