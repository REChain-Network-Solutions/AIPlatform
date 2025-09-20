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

namespace YooKassa\Request\PaymentMethods\ConfirmationData;

use YooKassa\Model\SavePaymentMethod\Confirmation\ConfirmationType;
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
 * @property bool $enforce Запрос на проведение платежа с аутентификацией по 3-D Secure. Будет работать, если оплату банковской картой вы по умолчанию принимаете без подтверждения платежа пользователем. В остальных случаях аутентификацией по 3-D Secure будет управлять ЮKassa. Если хотите принимать платежи без дополнительного подтверждения пользователем, напишите вашему менеджеру ЮKassa.
 * @property string $return_url URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов.
 * @property string $returnUrl URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице. Не более 2048 символов.
 * @property string $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь. Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)). Возможные значения: ru_RU, en_US. Регистр важен.
 */
class ConfirmationRedirect extends AbstractConfirmation
{
    /**
     * Запрос на проведение платежа с аутентификацией по 3-D Secure.
     *
     * @var bool|null
     */
    #[Assert\Type('bool')]
    private ?bool $_enforce = false;

    /**
     * URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 2048)]
    private ?string $_return_url = null;

    /**
     * Язык интерфейса, писем и смс, которые будет видеть или получать пользователь.
     * Формат соответствует [ISO/IEC 15897](https://en.wikipedia.org/wiki/Locale_(computer_software)).
     * Возможные значения: ru_RU, en_US. Регистр важен.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_locale = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(ConfirmationType::REDIRECT);
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
     * @param string|null $return_url URL, на который вернется пользователь после подтверждения или отмены платежа на веб-странице.
     *
     * @return self
     */
    public function setReturnUrl(?string $return_url = null): self
    {
        $this->_return_url = $this->validatePropertyValue('_return_url', $return_url);
        return $this;
    }

    /**
     * Возвращает locale.
     *
     * @return string|null
     */
    public function getLocale(): ?string
    {
        return $this->_locale;
    }

    /**
     * Устанавливает locale.
     *
     * @param string|null $locale Язык интерфейса, писем и смс, которые будет видеть или получать пользователь.
     *
     * @return self
     */
    public function setLocale(?string $locale = null): self
    {
        $this->_locale = $this->validatePropertyValue('_locale', $locale);
        return $this;
    }
}

