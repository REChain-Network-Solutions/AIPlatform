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

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель AuthorizationDetails.
 *
 * Данные об авторизации платежа при оплате банковской картой.
 *
 * Присутствуют только для этих способов оплаты: банковская карта, Mir Pay, SberPay, T-Pay.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $rrn Retrieval Reference Number — идентификатор банковской транзакции.
 * @property string $authСode Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.
 * @property string $auth_code Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.
 * @property ThreeDSecure $threeDSecure Данные о прохождении пользователем аутентификации по 3‑D Secure.
 * @property ThreeDSecure $three_d_secure Данные о прохождении пользователем аутентификации по 3‑D Secure.
 */
class AuthorizationDetails extends AbstractObject implements AuthorizationDetailsInterface
{
    /**
     * Retrieval Reference Number — идентификатор банковской транзакции.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_rrn = null;

    /**
     * Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_auth_code = null;

    /**
     * @var ThreeDSecure|null Данные о прохождении пользователем аутентификации по 3‑D Secure.
     */
    #[Assert\NotNull]
    #[Assert\Valid]
    #[Assert\Type(ThreeDSecure::class)]
    private ?ThreeDSecure $_three_d_secure = null;

    /**
     * Возвращает rrn.
     *
     * @return string|null
     */
    public function getRrn(): ?string
    {
        return $this->_rrn;
    }

    /**
     * Устанавливает rrn.
     *
     * @param string|null $rrn Retrieval Reference Number — идентификатор банковской транзакции.
     *
     * @return self
     */
    public function setRrn(?string $rrn = null): self
    {
        $this->_rrn = $this->validatePropertyValue('_rrn', $rrn);
        return $this;
    }

    /**
     * Возвращает auth_code.
     *
     * @return string|null Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.
     */
    public function getAuthCode(): ?string
    {
        return $this->_auth_code;
    }

    /**
     * Устанавливает auth_code.
     *
     * @param string|null $auth_code Код авторизации. Выдается эмитентом и подтверждает проведение авторизации.
     *
     * @return self
     */
    public function setAuthCode(?string $auth_code = null): self
    {
        $this->_auth_code = $this->validatePropertyValue('_auth_code', $auth_code);
        return $this;
    }

    /**
     * Возвращает three_d_secure.
     *
     * @return ThreeDSecure|null
     */
    public function getThreeDSecure(): ?ThreeDSecure
    {
        return $this->_three_d_secure;
    }

    /**
     * Устанавливает three_d_secure.
     *
     * @param ThreeDSecure|array|null $three_d_secure
     *
     * @return self
     */
    public function setThreeDSecure(mixed $three_d_secure = null): self
    {
        $this->_three_d_secure = $this->validatePropertyValue('_three_d_secure', $three_d_secure);
        return $this;
    }
}
