import axios from 'axios'

export const API_BASE = import.meta.env.VITE_API_URL || ''
export const api = axios.create({ baseURL: API_BASE, timeout: 30000 })

export function errorMessage(error) {
  const detail = error?.response?.data?.detail
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const messages = detail.map((issue) => {
      if (typeof issue === 'string') return issue
      const location = Array.isArray(issue?.loc)
        ? issue.loc.filter((part) => !['body', 'query', 'path'].includes(part)).join('.')
        : ''
      const message = issue?.msg || issue?.message
      return [location, message].filter(Boolean).join(': ')
    }).filter(Boolean)
    if (messages.length) return messages.join('; ')
  }
  if (detail && typeof detail === 'object') {
    const message = detail.message || detail.error
    if (typeof message === 'string') return message
  }
  return error?.message || 'Something went wrong. Please try again.'
}
