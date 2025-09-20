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

namespace YooKassa\Request\SelfEmployed;

use YooKassa\Common\AbstractRequest;
use YooKassa\Model\SelfEmployed\SelfEmployedConfirmation;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель SelfEmployedRequest.
 *
 * Запрос на создание объекта самозанятого.
 *
 * @example 02-builder.php 232 18 Пример использования билдера
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property null|string $itn ИНН самозанятого. Формат: 12 цифр без пробелов. Обязательный параметр, если не передан phone.
 * @property null|string $phone Телефон самозанятого, который привязан к личному кабинету в сервисе Мой налог.
 * @property null|SelfEmployedRequestConfirmation $confirmation Сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков в сервисе Мой налог.
 */
class SelfEmployedRequest extends AbstractRequest implements SelfEmployedRequestInterface
{
    /**
     * ИНН самозанятого.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/^\d{12}$/")]
    private ?string $_itn = null;

    /**
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Regex("/^[0-9]{4,15}$/")]
    private ?string $_phone = null;

    /**
     * @var SelfEmployedRequestConfirmation|null
     */
    #[Assert\Valid]
    #[Assert\Type(SelfEmployedRequestConfirmation::class)]
    private ?SelfEmployedRequestConfirmation $_confirmation = null;

    /**
     * Возвращает ИНН самозанятого.
     *
     * @return string|null ИНН самозанятого
     */
    public function getItn(): ?string
    {
        return $this->_itn;
    }

    /**
     * Устанавливает ИНН самозанятого.
     *
     * @param string|null $itn ИНН самозанятого
     *
     * @return self
     */
    public function setItn(?string $itn = null): self
    {
        $this->_itn = $this->validatePropertyValue('_itn', $itn);
        return $this;
    }

    /**
     * Проверяет наличие ИНН самозанятого в запросе.
     *
     * @return bool True если ИНН самозанятого задан, false если нет
     */
    public function hasItn(): bool
    {
        return !empty($this->_itn);
    }

    /**
     * Возвращает телефон самозанятого.
     *
     * @return string|null Телефон самозанятого
     */
    public function getPhone(): ?string
    {
        return $this->_phone;
    }

    /**
     * Устанавливает телефон самозанятого.
     *
     * @param string|null $phone Телефон самозанятого
     *
     * @return self
     */
    public function setPhone(?string $phone = null): self
    {
        $this->_phone = $this->validatePropertyValue('_phone', $phone);
        return $this;
    }

    /**
     * Проверяет наличие телефона самозанятого в запросе.
     *
     * @return bool True если телефон самозанятого задан, false если нет
     */
    public function hasPhone(): bool
    {
        return !empty($this->_phone);
    }

    /**
     * Возвращает сценарий подтверждения.
     *
     * @return SelfEmployedRequestConfirmation|null Сценарий подтверждения
     */
    public function getConfirmation(): ?SelfEmployedRequestConfirmation
    {
        return $this->_confirmation;
    }

    /**
     * Устанавливает сценарий подтверждения.
     *
     * @param null|array|SelfEmployedConfirmation $confirmation Сценарий подтверждения
     *
     * @return $this
     */
    public function setConfirmation(mixed $confirmation = null): self
    {
        if (is_array($confirmation)) {
            $confirmation = (new SelfEmployedRequestConfirmationFactory)->factoryFromArray($confirmation);
        }
        $this->_confirmation = $this->validatePropertyValue('_confirmation', $confirmation);
        return $this;
    }

    /**
     * Проверяет наличие сценария подтверждения самозанятого в запросе.
     *
     * @return bool True если сценарий подтверждения самозанятого задан, false если нет
     */
    public function hasConfirmation(): bool
    {
        return !empty($this->_confirmation);
    }

    /**
     * Проверяет на валидность текущий объект
     *
     * @return bool True если объект запроса валиден, false если нет
     */
    public function validate(): bool
    {
        if (!$this->hasPhone() && !$this->hasItn()) {
            $this->setValidationError('Both itn and phone values are empty in SelfEmployedRequest');

            return false;
        }

        return true;
    }

    /**
     * Возвращает билдер объектов запросов создания платежа.
     *
     * @return SelfEmployedRequestBuilder
     */
    public static function builder(): SelfEmployedRequestBuilder
    {
        return new SelfEmployedRequestBuilder();
    }
}
