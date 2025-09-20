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

namespace YooKassa\Request\Payments\ConfirmationAttributes;

use YooKassa\Model\Payment\ConfirmationType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ConfirmationAttributesRedirect.
 *
 * Сценарий при котором пользователю необходимо что-то сделать на странице ЮKassa или ее партнера
 * (например, ввести данные банковской карты или пройти аутентификацию по 3-D Secure).
 * Вам нужно перенаправить пользователя на confirmation_url, полученный в платеже.
 * При успешной оплате (и если что-то пойдет не так) ЮKassa вернет пользователя на return_url,
 * который вы отправите в запросе на создание платежа.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Код сценария подтверждения
 * @property string $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь
 * @property bool $enforce Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для
 *                         оплаты банковскими картами. По умолчанию определяется политикой платежной системы.
 * @property string $returnUrl URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера.
 * @property string $return_url URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера.
 */
class ConfirmationAttributesRedirect extends AbstractConfirmationAttributes
{
    /**
     * @var bool Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для оплаты банковскими картами. По умолчанию определяется политикой платежной системы.
     */
    #[Assert\Type('bool')]
    private ?bool $_enforce = null;

    /**
     * @var string|null URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 2048)]
    private ?string $_return_url = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(ConfirmationType::REDIRECT);
    }

    /**
     * @return bool|null Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для оплаты банковскими картами. По умолчанию определяется политикой платежной системы.
     */
    public function getEnforce(): ?bool
    {
        return $this->_enforce;
    }

    /**
     * @param bool|null $enforce Требование принудительного подтверждения платежа покупателем, требование 3-D Secure для оплаты банковскими картами. По умолчанию определяется политикой платежной системы.
     *
     * @return self
     */
    public function setEnforce(mixed $enforce = null): self
    {
        $this->_enforce = $this->validatePropertyValue('_enforce', $enforce);
        return $this;
    }

    /**
     * @return string|null URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера
     */
    public function getReturnUrl(): ?string
    {
        return $this->_return_url;
    }

    /**
     * @param string|null $return_url URL на который вернется плательщик после подтверждения или отмены платежа на странице партнера
     *
     * @return self
     */
    public function setReturnUrl(?string $return_url = null): self
    {
        $this->_return_url = $this->validatePropertyValue('_return_url', $return_url);
        return $this;
    }
}
