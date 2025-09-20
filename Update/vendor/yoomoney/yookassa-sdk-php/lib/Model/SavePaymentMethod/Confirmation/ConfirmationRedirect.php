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

namespace YooKassa\Model\SavePaymentMethod\Confirmation;

use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ConfirmationRedirect.
 *
 * Перенаправление пользователя на сайт ЮKassa для подтверждения привязки или страницу банка-эмитента для аутентификации по 3-D Secure.
 * 
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $confirmation_url URL, на который необходимо перенаправить пользователя для подтверждения привязки.
 * @property string $confirmationUrl URL, на который необходимо перенаправить пользователя для подтверждения привязки.
 * @property bool $enforce Запрос на проведение платежа с аутентификацией по 3-D Secure. Будет работать, если оплату банковской картой вы по умолчанию принимаете без подтверждения платежа пользователем. В остальных случаях аутентификацией по 3-D Secure будет управлять ЮKassa. Если хотите принимать платежи без дополнительного подтверждения пользователем, напишите вашему менеджеру ЮKassa.
 * @property string $return_url URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов.
 * @property string $returnUrl URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов.
*/
class ConfirmationRedirect extends AbstractConfirmation
{
    /**
     * URL, на который необходимо перенаправить пользователя для подтверждения привязки.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_confirmation_url = null;

    /**
     * Запрос на проведение платежа с аутентификацией по 3-D Secure.
     *
     * @var bool|null
     */
    #[Assert\Type('bool')]
    private ?bool $_enforce = false;

    /**
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_return_url = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(ConfirmationType::REDIRECT);
    }

    /**
     * Возвращает confirmation_url.
     *
     * @return string|null
     */
    public function getConfirmationUrl(): ?string
    {
        return $this->_confirmation_url;
    }

    /**
     * Устанавливает confirmation_url.
     *
     * @param string|null $confirmation_url URL, на который необходимо перенаправить пользователя для подтверждения привязки.
     *
     * @return self
     */
    public function setConfirmationUrl(?string $confirmation_url = null): self
    {
        $this->_confirmation_url = $this->validatePropertyValue('_confirmation_url', $confirmation_url);
        return $this;
    }

    /**
     * Возвращает enforce.
     *
     * @return bool|null
     */
    public function getEnforce(): ?bool
    {
        return $this->_enforce;
    }

    /**
     * Устанавливает enforce.
     *
     * @param bool|null $enforce Запрос на проведение платежа с аутентификацией по 3-D Secure.
     *
     * @return self
     */
    public function setEnforce(?bool $enforce = null): self
    {
        $this->_enforce = $this->validatePropertyValue('_enforce', $enforce);
        return $this;
    }

    /**
     * Возвращает return_url.
     *
     * @return string|null
     */
    public function getReturnUrl(): ?string
    {
        return $this->_return_url;
    }

    /**
     * Устанавливает return_url.
     *
     * @param string|null $return_url URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов.
     *
     * @return self
     */
    public function setReturnUrl(?string $return_url = null): self
    {
        $this->_return_url = $this->validatePropertyValue('_return_url', $return_url);
        return $this;
    }
}

